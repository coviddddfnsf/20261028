import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import re
import traceback
import os
import time
import numpy as np
import h5py
from argparse import ArgumentParser

from description.utils.generate_episode_instructions import generate_episode_descriptions

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def summarize_force(hdf5_path):
    """读取一个 episode 的 hdf5，统计工作臂（或双臂）的力与接触情况。

    返回 None 表示文件里没有力觉字段；返回 dict 时包含
    frames / force_dim / max_force / contact_frames 等，并保留
    左右臂分项字段以便双臂数据排查。
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            if "observations" not in f or "force" not in f["observations"]:
                return None
            force = np.asarray(f["observations"]["force"][()], dtype=np.float32)
            obs_group = f["observations"]
            contact = None
            for key in ("contact_gt", "contact"):
                if key in obs_group:
                    contact = np.asarray(obs_group[key][()], dtype=np.float32)
                    break
    except Exception as e:  # 统计失败不应影响采集主流程
        return {"error": str(e)}

    if force.size == 0:
        return None
    if force.ndim == 1:
        force = force.reshape(-1, 1)
    force_dim = int(force.shape[1])

    left_norm = np.linalg.norm(force[:, :3], axis=1) if force_dim >= 3 else np.zeros(len(force))
    # 双臂布局 [左 6 维, 右 6 维] 时才有右臂分量
    right_norm = np.linalg.norm(force[:, 6:9], axis=1) if force_dim >= 9 else None
    arm_norms = [left_norm] if right_norm is None else [left_norm, right_norm]
    max_force = max(float(nrm.max()) for nrm in arm_norms)
    mean_force = float(np.mean([nrm.mean() for nrm in arm_norms]))

    if contact is None:
        contact_frames = 0
        contact_frames_left = contact_frames_right = 0
    else:
        if contact.ndim == 1:
            contact = contact.reshape(-1, 1)
        aligned = np.zeros((len(force), contact.shape[1]), dtype=np.float32)
        rows = min(len(force), len(contact))
        aligned[:rows] = contact[:rows]
        contact_frames = int((aligned.max(axis=1) > 0.5).sum())
        contact_frames_left = int((aligned[:, 0] > 0.5).sum())
        contact_frames_right = (
            int((aligned[:, 1] > 0.5).sum()) if aligned.shape[1] > 1 else contact_frames_left
        )

    stats = {
        "frames": int(force.shape[0]),
        "force_dim": force_dim,
        "max_force": max_force,
        "mean_force": mean_force,
        "contact_frames": contact_frames,
        "contact_ratio": float(contact_frames / max(len(force), 1)),
        "max_force_left": float(left_norm.max()),
        "mean_force_left": float(left_norm.mean()),
        "contact_frames_left": contact_frames_left,
        "contact_ratio_left": float(contact_frames_left / max(len(force), 1)),
    }
    if right_norm is not None:
        stats.update(
            {
                "max_force_right": float(right_norm.max()),
                "mean_force_right": float(right_norm.mean()),
                "contact_frames_right": contact_frames_right,
                "contact_ratio_right": float(contact_frames_right / max(len(force), 1)),
            }
        )
    return stats


def record_force_stats(args, episode_idx):
    """把单个 episode 的力觉统计写入 save_path/force_info.json，并打印提示。"""
    hdf5_path = os.path.join(
        args["save_path"], "data", f"episode_{episode_idx:07d}.hdf5"
    )
    stats = summarize_force(hdf5_path)

    info_path = os.path.join(args["save_path"], "force_info.json")
    info_db = {}
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)
        except Exception:
            info_db = {}
    info_db[f"episode_{episode_idx}"] = stats
    with open(info_path, "w", encoding="utf-8") as file:
        json.dump(info_db, file, ensure_ascii=False, indent=4)

    if stats is None:
        print(f"\033[91m[Force] episode {episode_idx}: hdf5 中未找到力觉数据\033[0m")
    elif "error" in stats:
        print(f"\033[91m[Force] episode {episode_idx}: 统计失败 ({stats['error']})\033[0m")
    elif stats["max_force"] <= 1e-6:
        print(
            f"\033[93m[Force] episode {episode_idx}: 工作臂全程未检测到夹爪接触力，"
            f"请确认力觉采集是否正常\033[0m"
        )
    else:
        detail = (
            f"最大受力={stats['max_force']:.3f} N "
            f"(接触帧={stats['contact_frames']}/{stats['frames']})"
        )
        if "max_force_right" in stats:
            detail += (
                f", 左臂最大受力={stats['max_force_left']:.3f} N "
                f", 右臂最大受力={stats['max_force_right']:.3f} N"
            )
        print(
            f"\033[92m[Force] episode {episode_idx}: frames={stats['frames']}, "
            f"force_dim={stats['force_dim']}, {detail}\033[0m"
        )
    return stats


def build_episode_instructions(args, episode_info, episode_idx, seed):
    fallback = args["task_name"].replace("_", " ")
    info = episode_info.get("info", {}) if isinstance(episode_info, dict) else {}
    generated = generate_episode_descriptions(
        args["task_name"], [info], int(args.get("language_num") or 100), seed=seed
    )
    if generated:
        episode_descriptions = generated[0]
    else:
        episode_descriptions = {"seen": [], "unseen": []}

    episode_descriptions["episode_index"] = episode_idx
    for split in ("seen", "unseen"):
        if not episode_descriptions.get(split):
            episode_descriptions[split] = [fallback]

    instruction_dir = os.path.join(args["save_path"], "instruction")
    os.makedirs(instruction_dir, exist_ok=True)
    with open(
        os.path.join(instruction_dir, f"episode_{episode_idx:07d}.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "seed": seed,
                "seen": episode_descriptions["seen"],
                "unseen": episode_descriptions["unseen"],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return episode_descriptions


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(task_name=None, task_config=None, episode_num=None):

    task = class_decorator(task_name)
    config_path = os.path.join(CONFIGS_PATH, f"{task_config}.yml")

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    if episode_num is not None:
        if episode_num < 1:
            raise ValueError("--episode_num must be a positive integer")
        args["episode_num"] = episode_num

    args['task_name'] = task_name

    # 力觉数据采集开关：配置 data_type.force 缺省时默认开启
    args.setdefault("data_type", {})
    args["data_type"].setdefault("force", True)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mForce Data Config:\033[0m " + str(args["data_type"]["force"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    # Directory name follows the embodiment, e.g. aloha-agilex -> aloha_agilex,
    # ARX-X5 -> arx_x5, mixed dual-arm a+b -> a_b.
    embodiment_dir = re.sub(r"[^A-Za-z0-9_]+", "_", embodiment_name).lower()
    args["save_path"] = os.path.join(
        args["save_path"],
        args["task_config"],
        str(args["task_name"]),
        embodiment_dir,
    )
    run(task, args)


def run(TASK_ENV, args):
    epid, suc_num, fail_num, seed_list = 0, 0, 0, []

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")

    # =========== Collect Seed ===========
    os.makedirs(args["save_path"], exist_ok=True)

    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")
        args["need_plan"] = True

        if os.path.exists(os.path.join(args["save_path"], "seed.txt")):
            with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
                seed_list = file.read().split()
                if len(seed_list) != 0:
                    seed_list = [int(i) for i in seed_list]
                    suc_num = len(seed_list)
                    epid = max(seed_list) + 1
            print(f"Exist seed file, Start from: {epid} / {suc_num}")

        while suc_num < args["episode_num"]:
            try:
                TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
                TASK_ENV.play_once()

                if TASK_ENV.plan_success and TASK_ENV.check_success():
                    joints_ok, joint_excess = TASK_ENV.planned_joints_legal()
                    if not joints_ok:
                        print(
                            f"simulate data episode {suc_num} fail! (seed = {epid}) "
                            f"planned joints exceed limits (excess={joint_excess:.3f} rad)"
                        )
                        fail_num += 1
                    else:
                        print(f"simulate data episode {suc_num} success! (seed = {epid})")
                        seed_list.append(epid)
                        TASK_ENV.save_traj_data(suc_num)
                        suc_num += 1
                else:
                    print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                    fail_num += 1

                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
            except UnStableError as e:
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(0.3)
            except Exception as e:
                # stack_trace = traceback.format_exc()
                print(" -------------")
                print(f"simulate data episode {suc_num} fail! (seed = {epid})")
                print("Error: ", e)
                print(" -------------")
                fail_num += 1
                TASK_ENV.close_env()

                if args["render_freq"]:
                    TASK_ENV.viewer.close()
                time.sleep(1)

            epid += 1

            with open(os.path.join(args["save_path"], "seed.txt"), "w") as file:
                for sed in seed_list:
                    file.write("%s " % sed)

        print(f"\nComplete simulation, failed \033[91m{fail_num}\033[0m times / {epid} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        with open(os.path.join(args["save_path"], "seed.txt"), "r") as file:
            seed_list = file.read().split()
            seed_list = [int(i) for i in seed_list]

    # =========== Collect Data ===========

    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True

        clear_cache_freq = args["clear_cache_freq"]

        st_idx = 0

        def exist_hdf5(idx):
            file_path = os.path.join(
                args["save_path"], "data", f"episode_{idx:07d}.hdf5"
            )
            return os.path.exists(file_path)

        while exist_hdf5(st_idx):
            st_idx += 1

        for episode_idx in range(st_idx, args["episode_num"]):
            print(f"\033[34mTask name: {args['task_name']}\033[0m")

            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=seed_list[episode_idx], **args)

            traj_data = TASK_ENV.load_tran_data(episode_idx)
            args["left_joint_path"] = traj_data["left_joint_path"]
            args["right_joint_path"] = traj_data["right_joint_path"]
            TASK_ENV.set_path_lst(args)

            info_file_path = os.path.join(args["save_path"], "scene_info.json")

            if not os.path.exists(info_file_path):
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump({}, file, ensure_ascii=False)

            with open(info_file_path, "r", encoding="utf-8") as file:
                info_db = json.load(file)

            info = TASK_ENV.play_once()
            info_db[f"episode_{episode_idx}"] = info

            with open(info_file_path, "w", encoding="utf-8") as file:
                json.dump(info_db, file, ensure_ascii=False, indent=4)

            episode_descriptions = build_episode_instructions(
                args, info, episode_idx, seed_list[episode_idx]
            )
            TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
            TASK_ENV.merge_pkl_to_hdf5_video(
                instructions=episode_descriptions["seen"],
            )
            # 力觉数据校验：确认 hdf5 里存在 observations/force 并输出统计
            if args["data_type"].get("force", True):
                record_force_stats(args, episode_idx)
            TASK_ENV.remove_data_cache()
            assert TASK_ENV.check_success(), "Collect Error"


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--episode_num", type=int, default=None)
    parser = parser.parse_args()
    task_name = parser.task_name
    task_config = parser.task_config

    main(
        task_name=task_name,
        task_config=task_config,
        episode_num=parser.episode_num,
    )
