import os
import h5py
import numpy as np
import cv2
import argparse
import json

from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import pack_robot_state, get_robot_action_dim_info, decode_image_bit

# 力信号：单臂主导任务 6 维，双臂 12 维；接触真值统一 1 维
DEFAULT_FORCE_DIM = 6
CONTACT_DIM = 1

def data_transform(path, episode_num, load_data_dir, save_dir, robot_action_dim_info):
    begin = 0
    floders = os.listdir(path)
    assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for current_episode in range(episode_num):
        load_path = os.path.join(load_data_dir, f"data/episode_{current_episode:07d}.hdf5")
        data = load_hdf5(load_path)
        state_all = pack_robot_state(data, action_type, robot_action_dim_info, source_type="dataset", state_type="state")
        action_all = pack_robot_state(data, action_type, robot_action_dim_info, source_type="dataset", state_type="action")
        
        qpos = []
        actions = []
        forces = []
        contacts = []
        cam_head_list = []
        cam_left_wrist_list = []
        cam_right_wrist_list = []

        # 探测原始 hdf5 里是否有力/接触信号
        has_force = 'observations' in data and 'force' in data.get('observations', {})
        if not has_force:
            # 兼容 state/ 目录下可能存放力的情况
            has_force = 'force' in data.get('state', {})
        observations = data.get('observations', {})
        has_contact = 'contact_gt' in observations or 'contact' in observations

        for j in range(state_all.shape[0]):
            
            state, action = state_all[j], action_all[j]

            state = state.astype(np.float32)
            qpos.append(state)

            # === [力信号提取：单臂 6 维 / 双臂 12 维，维度稍后统一] ===
            force_frame = np.zeros(0, dtype=np.float32)
            if has_force:
                try:
                    if 'observations' in data and 'force' in data['observations']:
                        force_frame = np.asarray(
                            data['observations']['force'][j], dtype=np.float32
                        ).reshape(-1)
                    else:
                        force_frame = np.asarray(
                            data['state']['force'][j], dtype=np.float32
                        ).reshape(-1)
                except Exception:
                    force_frame = np.zeros(0, dtype=np.float32)
            forces.append(force_frame)

            # === [接触真值：来自物理碰撞检测，不做力阈值伪标注] ===
            contact = np.zeros(CONTACT_DIM, dtype=np.float32)
            if has_contact:
                try:
                    contact_key = (
                        'contact_gt' if 'contact_gt' in observations else 'contact'
                    )
                    c = np.asarray(
                        observations[contact_key][j], dtype=np.float32
                    ).reshape(-1)
                    if c.size:
                        contact[0] = float(c.max() > 0.5)
                except Exception:
                    pass
            contacts.append(contact)

            cam_head_bit = data['vision']["cam_head"]['colors'][j]
            cam_head = decode_image_bit(cam_head_bit)
            cam_head_resized = cv2.resize(cam_head, (640, 480))
            cam_head_list.append(cam_head_resized)

            cam_left_wrist_bit = data['vision']['cam_left_wrist']['colors'][j]
            cam_left_wrist =decode_image_bit(cam_left_wrist_bit)
            cam_left_wrist_resized = cv2.resize(cam_left_wrist, (640, 480))
            cam_left_wrist_list.append(cam_left_wrist_resized)

            cam_right_wrist_bit = data['vision']['cam_right_wrist']['colors'][j]
            cam_right_wrist = decode_image_bit(cam_right_wrist_bit)
            cam_right_wrist_resized = cv2.resize(cam_right_wrist, (640, 480))
            cam_right_wrist_list.append(cam_right_wrist_resized)

            actions.append(action)

        hdf5path = os.path.join(save_dir, f"episode_{current_episode}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions, dtype=np.float32))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos, dtype=np.float32))
            # === [力信号写入：(N, D) / 接触真值 (N, 1)] ===
            if has_force:
                force_dim = max([f.size for f in forces] + [DEFAULT_FORCE_DIM])
                force_arr = np.zeros((len(forces), force_dim), dtype=np.float32)
                for i, f in enumerate(forces):
                    cols = min(force_dim, f.size)
                    force_arr[i, :cols] = f[:cols]
                obs.create_dataset("force", data=force_arr)
                obs.attrs["force_dim"] = int(force_dim)
            if has_contact:
                obs.create_dataset(
                    "contact_gt", data=np.asarray(contacts, dtype=np.float32)
                )
            image = obs.create_group("images")
            image.create_dataset(
                "cam_head",
                data=np.stack(cam_head_list),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            image.create_dataset(
                "cam_left_wrist",
                data=np.stack(cam_left_wrist_list),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            image.create_dataset(
                "cam_right_wrist",
                data=np.stack(cam_right_wrist_list),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )

        begin += 1
        print(f"ACT: proccess episode {current_episode + 1} success!", end='\r')
    
    return begin

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument("bench_name", type=str, help="The name of the dataset (e.g., ACT)",)
    parser.add_argument("ckpt_name", type=str, help="Run name; also selects raw task dir under data/<bench>/",)
    parser.add_argument("env_cfg_type", type=str, help="The name of the environment config",)
    parser.add_argument("action_type", type=str, help="The type of action to process (e.g., joint)",)
    parser.add_argument(
        "--episode_num",
        type=int,
        default=None,
        help="Optional episode limit; only the first N episodes are converted.",
    )
    args = parser.parse_args()
    
    bench_name = args.bench_name
    ckpt_name = args.ckpt_name
    env_cfg_type = args.env_cfg_type
    action_type = args.action_type

    save_dir = f"processed_data/{bench_name}/{ckpt_name}/{env_cfg_type}-{action_type}"

    load_data_dir = os.path.join("../../../data", str(bench_name), str(ckpt_name), str(env_cfg_type))

    robot_action_dim_info = get_robot_action_dim_info(env_cfg_type)

    raw_episode_dir = os.path.join("../../../data/", bench_name, ckpt_name, env_cfg_type, 'data')
    available_episodes = len([
        name for name in os.listdir(raw_episode_dir)
        if name.startswith("episode_") and name.endswith(".hdf5")
    ])
    episode_num = args.episode_num if args.episode_num is not None else available_episodes
    if episode_num <= 0 or episode_num > available_episodes:
        raise ValueError(
            f"Requested {episode_num} episodes, but {available_episodes} are available."
        )

    begin = data_transform(raw_episode_dir, episode_num, load_data_dir, save_dir, robot_action_dim_info)

    print()

    TASK_CONFIGS_PATH = "./TASK_CONFIGS.json"

    try:
        with open(TASK_CONFIGS_PATH, "r") as f:
            TASK_CONFIGS = json.load(f)
    except Exception:
        TASK_CONFIGS = {}

    TASK_CONFIGS[f"{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"] = {
        "dataset_dir": save_dir,
        "num_episodes": episode_num,
        "episode_len": 5000,
        "camera_names": ["cam_head", "cam_right_wrist", "cam_left_wrist"],
    }

    with open(TASK_CONFIGS_PATH, "w") as f:
        json.dump(TASK_CONFIGS, f, indent=4)
