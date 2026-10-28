import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except:
    from .main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):

    def __init__(self, args_override, model_cfg=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, model_cfg)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.gate_type = args_override.get(
            "gate_type", getattr(model_cfg, "gate_type", "none")
        )
        self.disable_force = bool(args_override.get("disable_force", False))

    def __call__(
        self, qpos, image, actions=None, is_pad=None, force=None,
        contact_gt=None, force_seq=None,
    ):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if self.disable_force:
            force = None
            force_seq = None
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad,
                force=force, force_seq=force_seq,
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]

            # === [门控辅助损失] ===
            loss_gate = torch.tensor(0.0, device=qpos.device)
            alpha = getattr(self.model, 'latest_alpha', None)
            if self.gate_type == "soft" and alpha is not None and contact_gt is not None:
                contact_gt_t = contact_gt.to(alpha.device).float()
                # 接触真值可能是 [左臂, 右臂]，取 OR 并集与门控系数对齐
                if contact_gt_t.dim() > 1 and contact_gt_t.shape[-1] > 1:
                    contact_gt_t = contact_gt_t.amax(dim=-1, keepdim=True)
                else:
                    contact_gt_t = contact_gt_t.view(-1, 1)
                loss_gate = F.binary_cross_entropy(alpha, contact_gt_t)
                loss_dict['loss_gate'] = loss_gate

            gate_loss_weight = 0.05
            loss_dict["loss"] = (
                loss_dict["l1"]
                + loss_dict["kl"] * self.kl_weight
                + gate_loss_weight * loss_gate
            )
            return loss_dict
        else:  # inference time
            # 即使没传 force (纯视觉评测)，自适应退化运行
            a_hat, _, (_, _) = self.model(
                qpos, image, env_state, force=force, force_seq=force_seq
            )
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT:

    def __init__(self, args_override=None, model_cfg=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }

        self.camera_names = model_cfg.camera_names

        # 先读取数据集统计，确保模型力维度与训练时一致
        ckpt_dir = args_override.get("ckpt_dir", "")
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl") if ckpt_dir else ""
        ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt") if ckpt_dir else ""
        if not ckpt_dir:
            raise ValueError("ACT requires ckpt_dir during evaluation.")
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"ACT checkpoint directory not found: {ckpt_dir}")
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"ACT dataset stats not found: {stats_path}")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        # 力维度以训练数据统计为准（单臂 6 维 / 双臂 12 维）
        self.force_dim = int(
            self.stats.get("force_dim", getattr(model_cfg, "force_dim", 12))
        )
        model_cfg.force_dim = self.force_dim
        self.disable_force = bool(getattr(model_cfg, "disable_force", False))
        self.gate_type = getattr(model_cfg, "gate_type", "none")
        self.force_history = []
        self.force_history_size = int(args_override["chunk_size"])

        self.policy = ACTPolicy(args_override, model_cfg)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args_override.get("temporal_agg", False)
        self.num_queries = args_override["chunk_size"]
        self.state_dim = model_cfg.action_dim  # Standard joint dimension for bimanual robot
        self.max_timesteps = 5000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)

        self.t = 0  # Current timestep

        # Load policy weights. Missing files are fatal because running with a
        # randomly initialized ACT policy is invalid.
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ACT policy checkpoint not found: {ckpt_path}")
        self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        
        self.obs_cache = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    @staticmethod
    def _fit_force_stat(values, dim, fill=0.0):
        """把统计量补齐/截断到 dim 维。"""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        fixed = np.full(dim, fill, dtype=np.float32)
        cols = min(dim, values.size)
        fixed[:cols] = values[:cols]
        return fixed

    def get_gate_alpha(self):
        """返回最近一次前向的门控系数，形状 [B]；无门控时返回 None。"""
        model = getattr(self.policy, "model", None)
        alpha = getattr(model, "latest_alpha", None)
        if alpha is None:
            return None
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().numpy()
        return np.asarray(alpha, dtype=np.float32).reshape(-1)

    def get_force_pred(self):
        """返回最近一次前向的力预测序列；当前模型未输出时返回 None。"""
        model = getattr(self.policy, "model", None)
        pred = getattr(model, "latest_force_pred", None)
        if pred is None:
            return None
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 3) if pred.size % 3 == 0 else pred.reshape(-1, 1)
        elif pred.ndim > 2:
            pred = pred.reshape(pred.shape[0], -1)
        # 统一取三维接触力，与评测指标中的 f_t ∈ R^3 对齐
        return pred[:, :3]

    def update_obs(self, obs):
        self.obs_cache = obs

    def get_action(self):
        obs = self.obs_cache

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images following imitate_episodes.py pattern
        # Stack images from all cameras
        curr_images = []
        for cam_name in self.camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        # === [新增：读取力信号，缺省时为 None] ===
        force_tensor = None
        force_seq_tensor = None
        if not self.disable_force and "force" in obs and obs["force"] is not None:
            force_np = np.array(obs["force"], dtype=np.float32).reshape(-1)
            # 与训练保持一致：单臂 6 维工作臂，双臂 12 维
            fixed_force = np.zeros(self.force_dim, dtype=np.float32)
            cols = min(self.force_dim, force_np.size)
            fixed_force[:cols] = force_np[:cols]
            force_np = fixed_force
            # 用 stats 里的 force_mean/std 归一化
            f_mean = np.asarray(
                self.stats.get("force_mean", np.zeros(self.force_dim, np.float32)),
                np.float32,
            ).reshape(-1)
            f_std = np.asarray(
                self.stats.get("force_std", np.ones(self.force_dim, np.float32)),
                np.float32,
            ).reshape(-1)
            f_mean = self._fit_force_stat(f_mean, self.force_dim)
            f_std = np.clip(
                self._fit_force_stat(f_std, self.force_dim, fill=1.0), 1e-2, np.inf
            )
            force_normalized = (force_np - f_mean) / f_std
            force_tensor = torch.from_numpy(force_normalized).float().to(self.device).unsqueeze(0)
            self.force_history.append(force_normalized)
            self.force_history = self.force_history[-self.force_history_size :]
            history = np.zeros(
                (self.force_history_size, self.force_dim), dtype=np.float32
            )
            history[-len(self.force_history) :] = np.stack(self.force_history)
            force_seq_tensor = torch.from_numpy(history).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(
                    qpos, curr_image, force=force_tensor,
                    force_seq=force_seq_tensor,
                )

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)

        self.t += 1
        return action