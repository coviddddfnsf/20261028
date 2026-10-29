import cv2
import numpy as np
import torch
from .detr.act_policy import ACT
from argparse import Namespace

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
import os

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        self.camera_names = model_cfg.get('camera_names', [])
        model_cfg['camera_names'] = self.camera_names

        self.model = self.get_model(model_cfg=model_cfg)
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        self.action_type = model_cfg['action_type']

    def get_model(self, model_cfg):
        if not model_cfg.get('ckpt_dir'):
            if not model_cfg.get('ckpt_name'):
                raise ValueError("ACT requires ckpt_name or ckpt_dir during evaluation.")
            # ckpt_name is the full run directory name under checkpoints/.
            model_cfg['ckpt_dir'] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'checkpoints', str(model_cfg['ckpt_name']))
        return ACT(model_cfg, Namespace(**model_cfg))

    def update_obs(self, obs):
        encoded_obs = self.encode_obs(obs, self.action_type, self.robot_action_dim_info)
        self.model.update_obs(encoded_obs)

    def get_gate_alpha(self):
        """门控系数，供评测脚本记录 alpha 曲线。"""
        getter = getattr(self.model, "get_gate_alpha", None)
        return getter() if callable(getter) else None

    def get_force_pred(self):
        """未来力预测序列，供评测脚本计算力预测 MAE 与曲线。"""
        getter = getattr(self.model, "get_force_pred", None)
        return getter() if callable(getter) else None

    def get_step_diagnostics(self):
        """单步诊断量：门控权重 / 动作块反事实修正 / 是否重规划。"""
        getter = getattr(self.model, "get_step_diagnostics", None)
        return getter() if callable(getter) else None
    
    # def update_obs_batch(self, obs_list): # TODO
    #     pass
    
    def get_action(self):
        actions = self.model.get_action()
        action_list = unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type='obs')
        return action_list

    # def get_action_batch(self, env_idx_list): # TODO
    #     pass

    def reset(self):
        self.model.force_history.clear()
        reset_counterfactual = getattr(self.model, "reset_counterfactual_state", None)
        if callable(reset_counterfactual):
            reset_counterfactual()
        # Reset temporal aggregation state if enabled
        if self.model.temporal_agg:
            self.model.all_time_actions = torch.zeros([
                self.model.max_timesteps,
                self.model.max_timesteps + self.model.num_queries,
                self.model.state_dim,
            ]).to(self.model.device)
            self.model.t = 0
        else:
            self.model.t = 0

    def encode_obs(self, observation, action_type, robot_action_dim_info):
        res_dict = dict()

        for camera_name in self.camera_names:
            if camera_name not in observation["vision"]:
                raise ValueError(f"Expected camera '{camera_name}' not found in observation['vision']")
            color = cv2.resize(observation["vision"][camera_name]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
            color = np.moveaxis(color, -1, 0) / 255.0
            res_dict[camera_name] = color
        
        res_dict["qpos"] = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")

        # 力觉/接触观测：单臂 6 维工作臂，双臂 12 维，与训练数据保持一致
        state = observation.get("state", {})
        force = state.get("force", observation.get("force"))
        if force is not None:
            res_dict["force"] = np.asarray(force, dtype=np.float32).reshape(-1)
        contact = state.get("contact_gt", observation.get("contact_gt"))
        if contact is None:
            contact = state.get("contact", observation.get("contact"))
        if contact is not None:
            res_dict["contact"] = np.asarray(contact, dtype=np.float32).reshape(-1)

        return res_dict