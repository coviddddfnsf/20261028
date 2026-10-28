import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader

import IPython

e = IPython.embed


# 单臂主导任务 6 维 [Fx,Fy,Fz,Tx,Ty,Tz]，双臂采集为 12 维
DEFAULT_FORCE_DIM = 6
# 接触真值固定 1 维：单臂为工作臂接触，双臂为任一侧接触（OR）
CONTACT_DIM = 1


def fit_force_dim(values, dim=None):
    """把单帧/多帧力信号整理成 (N, dim)；dim 为 None 时保持原维度。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if dim is None or values.shape[1] == dim:
        return values
    fixed = np.zeros((values.shape[0], dim), dtype=np.float32)
    cols = min(dim, values.shape[1])
    fixed[:, :cols] = values[:, :cols]
    return fixed


def fit_contact_dim(values):
    """接触真值整理为长度 1 的向量；双臂多值时取 OR。"""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(CONTACT_DIM, dtype=np.float32)
    return np.array([float(values.max() > 0.5)], dtype=np.float32)


class EpisodicDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        max_action_len,
        chunk_size=None,
    ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_action_len = max_action_len
        # force_seq_gt 的长度与模型一次预测的 chunk 对齐
        self.chunk_size = int(chunk_size or max_action_len)
        self.is_sim = None
        self._force_dim = int(norm_stats.get("force_dim", DEFAULT_FORCE_DIM))
        self._has_contact = False
        # 探测 hdf5 里是否有 force 字段（第一帧打开一次即可）
        self._has_force = self._detect_force()
        self.__getitem__(0)  # initialize self.is_sim

    def _detect_force(self):
        if len(self.episode_ids) == 0:
            return False
        probe_path = os.path.join(self.dataset_dir, f"episode_{self.episode_ids[0]}.hdf5")
        if not os.path.isfile(probe_path):
            return False
        try:
            with h5py.File(probe_path, "r") as root:
                if "/observations/force" not in root:
                    return False
                shape = root["/observations/force"].shape
                if len(shape) > 1 and shape[1] > 0:
                    self._force_dim = int(shape[1])
                self._has_contact = (
                    "/observations/contact_gt" in root or "/observations/contact" in root
                )
                return True
        except Exception:
            return False

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            is_sim = None
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            # The force window ends at the sampled observation, matching online inference.
            force = np.zeros(self._force_dim, dtype=np.float32)
            contact_gt = np.zeros(CONTACT_DIM, dtype=np.float32)
            force_seq_gt = np.zeros((self.chunk_size, self._force_dim), dtype=np.float32)
            if self._has_force:
                force_all = fit_force_dim(
                    root["/observations/force"][()], self._force_dim
                )
                force = force_all[start_ts]

                seq_start = max(0, start_ts - self.chunk_size + 1)
                seq = force_all[seq_start : start_ts + 1]
                force_seq_gt[-len(seq) :] = seq

                if self._has_contact:
                    contact_all = None
                    for contact_key in (
                        "/observations/contact_gt",
                        "/observations/contact",
                    ):
                        if contact_key in root:
                            contact_all = np.asarray(root[contact_key][()], dtype=np.float32)
                            break
                    if contact_all is not None:
                        contact_all = contact_all.reshape(len(contact_all), -1).max(axis=1)
                        if start_ts < len(contact_all):
                            contact_gt = fit_contact_dim(contact_all[start_ts])

            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1):]  # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.ones(self.max_action_len, dtype=bool)
        is_pad[:action_len] = 0

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        force_data = torch.from_numpy(force).float()
        contact_data = torch.from_numpy(contact_gt).float()
        force_seq_data = torch.from_numpy(force_seq_gt).float()

        # channel last
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        # force 归一化（维度自适应：单臂 6 维 / 双臂 12 维）
        if self._has_force:
            f_mean = torch.from_numpy(
                fit_force_dim(
                    self.norm_stats.get(
                        "force_mean", np.zeros(self._force_dim, dtype=np.float32)
                    ),
                    self._force_dim,
                )[0]
            ).float()
            f_std = torch.from_numpy(
                fit_force_dim(
                    self.norm_stats.get(
                        "force_std", np.ones(self._force_dim, dtype=np.float32)
                    ),
                    self._force_dim,
                )[0]
            ).float()
            f_std = torch.clamp(f_std, 1e-2, float("inf"))
            force_data = (force_data - f_mean) / f_std
            force_seq_data = (force_seq_data - f_mean) / f_std

        return (
            image_data,
            qpos_data,
            action_data,
            is_pad,
            force_data,
            contact_data,
            force_seq_data,
        )


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    all_force_data = []
    has_force = False
    force_dim = None
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]  # Assuming this is a numpy array
            action = root["/action"][()]
            if "/observations/force" in root:
                force = fit_force_dim(root["/observations/force"][()])
                if force.size:
                    all_force_data.append(force)
                    has_force = True
                    current_dim = int(force.shape[1])
                    force_dim = current_dim if force_dim is None else max(force_dim, current_dim)
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # Pad all tensors to the maximum size
    max_qpos_len = max(q.size(0) for q in all_qpos_data)
    max_action_len = max(a.size(0) for a in all_action_data)

    padded_qpos = []
    for qpos in all_qpos_data:
        current_len = qpos.size(0)
        if current_len < max_qpos_len:
            # Pad with the last element
            pad = qpos[-1:].repeat(max_qpos_len - current_len, 1)
            qpos = torch.cat([qpos, pad], dim=0)
        padded_qpos.append(qpos)

    padded_action = []
    for action in all_action_data:
        current_len = action.size(0)
        if current_len < max_action_len:
            pad = action[-1:].repeat(max_action_len - current_len, 1)
            action = torch.cat([action, pad], dim=0)
        padded_action.append(action)

    all_qpos_data = torch.stack(padded_qpos)
    all_action_data = torch.stack(padded_action)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }

    # force normalization（仅当 hdf5 里有 force 时）
    force_dim = force_dim or DEFAULT_FORCE_DIM
    stats["force_dim"] = int(force_dim)
    if has_force and len(all_force_data) > 0:
        all_force = torch.cat(
            [torch.from_numpy(fit_force_dim(f, force_dim)) for f in all_force_data],
            dim=0,
        )  # [total_T, force_dim]
        force_mean = all_force.mean(dim=0)
        force_std = all_force.std(dim=0).clamp(1e-2, np.inf)
        stats["force_mean"] = force_mean.numpy()
        stats["force_std"] = force_std.numpy()
        print(
            f"[force] 检测到力数据 (dim={force_dim})，"
            f"force_mean={force_mean.numpy().round(3)}, "
            f"force_std={force_std.numpy().round(3)}"
        )
    else:
        stats["force_mean"] = np.zeros(force_dim, dtype=np.float32)
        stats["force_std"] = np.ones(force_dim, dtype=np.float32)
        print("[force] hdf5 中未检测到力数据，将使用零向量（纯视觉 ACT 模式）")

    return stats, max_action_len


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    chunk_size=None,
):
    print(f"\nData from: {dataset_dir}\n")
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats, max_action_len = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices, dataset_dir, camera_names, norm_stats, max_action_len, chunk_size
    )
    val_dataset = EpisodicDataset(
        val_indices, dataset_dir, camera_names, norm_stats, max_action_len, chunk_size
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
