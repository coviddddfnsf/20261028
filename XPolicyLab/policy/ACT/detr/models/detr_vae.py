# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np

import IPython
import os

e = IPython.embed

# ==================== 创新模块定义区 ====================
# [方案二 / Ours 主推]: 接触感知自适应软门控
class ContactAwareSoftGate(nn.Module):
    def __init__(self, visual_dim, proprio_dim=14, force_dim=12, hidden_dim=128):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(visual_dim + proprio_dim + force_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 严格将门控系数压缩至 0~1
        )
        self.force_proj = nn.Linear(force_dim, visual_dim)

    def forward(self, visual_global, qpos, force):
        feat_cat = torch.cat([visual_global.detach(), qpos, force], dim=-1)
        alpha = self.gate_mlp(feat_cat)  # [B, 1]
        gated_force = alpha * self.force_proj(force)  # [B, visual_dim]
        return gated_force, alpha


# [方案三]: 跨频率多尺度时序聚合与门控交叉注意力
class MultiRateCrossGate(nn.Module):
    def __init__(self, hidden_dim, temporal_window=2, nheads=4, force_dim=12):
        super().__init__()
        self.temporal_window = temporal_window
        self.force_proj = nn.Linear(force_dim, hidden_dim)
        self.temporal_convs = nn.ModuleList(
            nn.Conv1d(force_dim, hidden_dim, kernel_size=size, padding=size // 2)
            for size in (3, 5, 7)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nheads, batch_first=False)
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens, force_seq):
        # force_seq: [B, T, force_dim]; preserve each sensor rate at multiple scales.
        force_seq = force_seq[:, -self.temporal_window :]
        force_tokens = self.force_proj(force_seq)
        conv_input = force_seq.transpose(1, 2)
        multi_rate_tokens = [
            conv(conv_input).transpose(1, 2) for conv in self.temporal_convs
        ]
        query = torch.cat([force_tokens, *multi_rate_tokens], dim=1).transpose(0, 1)
        attn_out, _ = self.cross_attn(
            query=query, key=visual_tokens, value=visual_tokens
        )
        gate = self.gate_fc(torch.cat([query, attn_out], dim=-1))
        fused_tokens = self.norm(query + gate * attn_out)
        return fused_tokens.mean(dim=0), gate.mean(dim=0)


# [方案四]: 双向自适应置信度路由门控 (视触互补)
class BiDirectionalGate(nn.Module):
    def __init__(self, visual_dim, proprio_dim=14, force_dim=12):
        super().__init__()
        self.vis_occ_head = nn.Sequential(
            nn.Linear(visual_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.force_contact_head = nn.Sequential(
            nn.Linear(force_dim + proprio_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.force_proj = nn.Linear(force_dim, visual_dim)

    def forward(self, visual_global, qpos, force):
        # 视觉置信度 beta (保底 0.2)，力置信度 alpha
        beta = 0.2 + 0.8 * self.vis_occ_head(visual_global)
        alpha = self.force_contact_head(torch.cat([force, qpos], dim=-1))
        gated_feat = self.force_proj(force) * alpha  # [B, D]
        return gated_feat, alpha, beta


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, gate_type='none', force_dim=12):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            gate_type: 门控类型, 'none'|'hard'|'soft'|'cross_attn'|'bidirectional'
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1 + 1 + num_queries,
                                                                      hidden_dim))  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(3, hidden_dim)  # learned position embed: [latent, proprio, force]

        # === [新增：统一门控架构模块] ===
        self.gate_type = gate_type  # 'none'|'hard'|'soft'|'cross_attn'|'bidirectional'
        # 单臂主导任务 6 维，双臂 12 维，由数据集统计信息决定
        self.force_dim = int(force_dim)
        # 方案一/Baseline2 基础投影（暴力硬拼接 & 差分硬门控共用）
        self.force_naive_proj = nn.Linear(self.force_dim, hidden_dim)
        # 方案二 软门控
        self.soft_gate = ContactAwareSoftGate(hidden_dim, proprio_dim=state_dim, force_dim=self.force_dim)
        # 方案三 交叉注意力门控
        self.cross_gate = MultiRateCrossGate(
            hidden_dim, temporal_window=num_queries, force_dim=self.force_dim
        )
        # 方案四 双向门控
        self.bidi_gate = BiDirectionalGate(hidden_dim, proprio_dim=state_dim, force_dim=self.force_dim)
        # 缓存门控系数，供 policy.py 读取辅助损失，避免改 forward 返回签名
        self.latest_alpha = None
        self.latest_beta = None

    def forward(
        self, qpos, image, env_state, actions=None, is_pad=None, force=None,
        force_seq=None,
    ):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        force: batch, force_dim  (可选, 6 维工作臂 / 12 维双臂力/力矩)
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        self.latest_alpha = None
        self.latest_beta = None
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)

            # === [门控多分支总调度] ===
            force_input = None
            if force is not None:
                force_clipped = 10.0 * torch.tanh(force / 10.0)
                if self.gate_type == 'none':
                    # Baseline 2: 暴力硬拼接 (直接投影输入)
                    force_input = self.force_naive_proj(force_clipped)
                elif self.gate_type == 'hard':
                    # Activate on either sustained contact force or a sharp force change.
                    f_norm = torch.norm(force_clipped[:, :3], dim=-1)
                    if force_clipped.shape[-1] >= 12:
                        f_norm = f_norm + torch.norm(force_clipped[:, 6:9], dim=-1)
                    delta_norm = torch.zeros_like(f_norm)
                    if force_seq is not None and force_seq.shape[1] > 1:
                        delta = force_seq[:, -1] - force_seq[:, -2]
                        delta_norm = torch.norm(delta[:, :3], dim=-1)
                        if delta.shape[-1] >= 12:
                            delta_norm = delta_norm + torch.norm(delta[:, 6:9], dim=-1)
                    gate_mask = ((f_norm > 0.5) | (delta_norm > 0.25)).float().unsqueeze(-1)
                    force_input = self.force_naive_proj(force_clipped) * gate_mask
                    self.latest_alpha = gate_mask
                elif self.gate_type == 'soft':
                    # 方案二: 可学习自适应软门控 (主推创新)
                    force_soft_clipped = 3.0 * torch.tanh(force / 3.0)
                    # 视觉全局均值
                    with torch.no_grad():
                        vis_flat = torch.cat([f.flatten(2).permute(2, 0, 1) for f in all_cam_features], dim=0)
                    vis_global = vis_flat.mean(dim=0)  # [B, D]
                    gated_feat, alpha = self.soft_gate(
                        vis_global, qpos, force_soft_clipped
                    )
                    force_input = gated_feat
                    self.latest_alpha = alpha
                elif self.gate_type == 'cross_attn':
                    # 方案三: 交叉注意力门控
                    visual_tokens = torch.cat([f.flatten(2).permute(2, 0, 1) for f in all_cam_features], dim=0)
                    if force_seq is None:
                        force_seq = force_clipped.unsqueeze(1)
                    safe_force_seq = 10.0 * torch.tanh(force_seq / 10.0)
                    fused_feat, alpha = self.cross_gate(visual_tokens, safe_force_seq)
                    force_input = fused_feat
                    self.latest_alpha = alpha
                elif self.gate_type == 'bidirectional':
                    # 方案四: 双向置信度门控
                    vis_flat = torch.cat([f.flatten(2).permute(2, 0, 1) for f in all_cam_features], dim=0)
                    vis_global = vis_flat.mean(dim=0)
                    gated_feat, alpha, beta = self.bidi_gate(
                        vis_global, qpos, force_clipped
                    )
                    force_input = gated_feat
                    self.latest_alpha = alpha
                    self.latest_beta = beta
                    # 视觉通道衰减 (beta 是 [B,1]，需要广播到视觉 token 维度)
                    # 注意: 这里不修改 src 本身，避免破坏 backbone 输出;
                    # 双向门控的视觉衰减通过 latest_beta 暴露给 policy 做辅助监督即可。
                else:
                    force_input = self.force_naive_proj(force_clipped)

            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input,
                                  self.additional_pos_embed.weight, force_input=force_input)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class CNNMLP(nn.Module):

    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim)  # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                                          nn.Conv2d(128, 64, kernel_size=5), nn.Conv2d(64, 32, kernel_size=5))
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + int(os.environ.get("ACT_ACTION_DIM"))
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=state_dim, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]  # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) 
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = int(os.environ.get("ACT_ACTION_DIM"))
    gate_type = getattr(args, 'gate_type', 'none')
    force_dim = int(getattr(args, 'force_dim', 12))

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
        gate_type=gate_type,
        force_dim=force_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model


def build_cnnmlp(args):
    state_dim = int(os.environ.get("ACT_ACTION_DIM"))

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model
