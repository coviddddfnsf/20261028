import os

# Set rendering backend for MuJoCo
os.environ["MUJOCO_GL"] = "egl"

import torch
import numpy as np
import pickle
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from utils import load_data  # data functions
from utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from detr.act_policy import ACTPolicy, CNNMLPPolicy

import IPython

e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    ckpt_setting = args["ckpt_setting"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]

    # get task parameters
    from constants import TASK_CONFIGS

    task_config = TASK_CONFIGS[ckpt_setting]
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]

    # fixed parameters
    state_dim = int(os.environ.get("ACT_ACTION_DIM"))
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        gate_type = args.get("gate_type", "none")
        disable_force = bool(args.get("disable_force", False))
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "gate_type": gate_type,
            "disable_force": disable_force,
        }
        print(f"[gate_type] 使用门控方案: {gate_type}; disable_force={disable_force}")
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "grad_accum_steps": args.get("grad_accum_steps", 1),
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "ckpt_setting": ckpt_setting,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "save_freq": args['save_freq']
    }

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        chunk_size=args["chunk_size"],
    )
    # 力信号维度由数据集决定：单臂 6 维 / 双臂 12 维
    policy_config["force_dim"] = int(stats.get("force_dim", 6))
    print(f"[force] 模型输入力维度: {policy_config['force_dim']}")

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    # ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    # torch.save(best_state_dict, ckpt_path)
    # print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def forward_pass(data, policy):
    (
        image_data,
        qpos_data,
        action_data,
        is_pad,
        force_data,
        contact_data,
        force_seq_data,
    ) = data
    image_data, qpos_data, action_data, is_pad, force_data, contact_data = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
        force_data.cuda(),
        contact_data.cuda(),
    )
    force_seq_data = force_seq_data.cuda()
    return policy(
        qpos_data,
        image_data,
        action_data,
        is_pad,
        force=force_data,
        force_seq=force_seq_data,
        contact_gt=contact_data,
    )


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    grad_accum_steps = config["grad_accum_steps"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "

        # training
        policy.train()
        optimizer.zero_grad()
        num_batches = len(train_dataloader)
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            group_start = (batch_idx // grad_accum_steps) * grad_accum_steps
            group_size = min(grad_accum_steps, num_batches - group_start)
            (forward_dict["loss"] / group_size).backward()
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx + 1 == num_batches:
                optimizer.step()
                optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "

        if (epoch + 1) % config['save_freq'] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
    # torch.save(best_state_dict, ckpt_path)
    # print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

    return best_ckpt_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument(
        "--policy_class",
        action="store",
        type=str,
        help="policy_class, capitalize",
        required=True,
    )
    parser.add_argument("--ckpt_setting", action="store", type=str, help="ckpt_setting", required=True)
    parser.add_argument("--batch_size", action="store", type=int, help="batch_size", default=16)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)
    parser.add_argument("--num_epochs", action="store", type=int, help="num_epochs", required=True)
    parser.add_argument("--lr", action="store", type=float, help="lr", required=True)

    # for ACT
    parser.add_argument("--kl_weight", action="store", type=int, help="KL Weight", required=False)
    parser.add_argument("--chunk_size", action="store", type=int, help="chunk_size", required=False)
    parser.add_argument("--hidden_dim", action="store", type=int, help="hidden_dim", required=False)
    parser.add_argument("--save_freq", action="store", type=int, help="save ckpt frequency", required=False, default=6000)
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        type=int,
        help="dim_feedforward",
        required=False,
    )
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument(
        "--gate_type",
        action="store",
        type=str,
        default="none",
        choices=["none", "hard", "soft", "cross_attn", "bidirectional"],
        help="门控方案: none=纯视觉ACT/baseline2, hard=硬门控, soft=软门控(主推), cross_attn=交叉注意力, bidirectional=双向门控",
    )
    parser.add_argument(
        "--disable_force",
        action="store_true",
        help="Disable force input (used to evaluate the visual-only baseline).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of micro-batches per optimizer step.",
    )

    parsed_args = vars(parser.parse_args())
    if parsed_args["grad_accum_steps"] < 1:
        parser.error("--grad_accum_steps must be a positive integer")
    main(parsed_args)
