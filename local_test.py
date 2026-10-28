"""
本地 Windows CPU 防爆自检：5 种 gate_type 全流程前向+反向+推理退化测试。
直接实例化 DETRVAE，绕过 build_ACT_model_and_optimizer 里的 .cuda() 调用。
"""
import os
import sys
import types
import torch

# 关键：build() 里读 os.environ["ACT_ACTION_DIM"]
os.environ["ACT_ACTION_DIM"] = "14"

HERE = os.path.dirname(os.path.abspath(__file__))
DETR_DIR = os.path.join(HERE, "XPolicyLab", "policy", "ACT", "detr")
sys.path.insert(0, DETR_DIR)

from models.detr_vae import build  # noqa: E402


def make_args(gate_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        hidden_dim=256,            # 小一点，CPU 快
        dim_feedforward=512,
        backbone="resnet18",
        lr_backbone=1e-5,
        dilation=False,
        position_embedding="sine",
        enc_layers=2,
        dec_layers=2,
        nheads=4,
        dropout=0.1,
        pre_norm=False,
        masks=False,
        chunk_size=50,
        camera_names=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        gate_type=gate_type,
    )


def run():
    schemes = ["none", "hard", "soft", "cross_attn", "bidirectional"]
    bs = 2
    seq = 50

    qpos = torch.randn(bs, 14)
    image = torch.randn(bs, 3, 3, 120, 160)   # 3 个视角
    force = torch.randn(bs, 12)
    actions = torch.randn(bs, seq, 14)
    is_pad = torch.zeros(bs, seq, dtype=torch.bool)
    contact_gt = torch.tensor([[0.0, 0.0], [1.0, 0.0]])

    print("开始本地 Windows CPU 5 方案防爆测试...\n")
    for s in schemes:
        args = make_args(s)
        model = build(args)
        model.train()

        # 1. 训练前向
        a_hat, is_pad_hat, (mu, logvar) = model(
            qpos, image, None, actions, is_pad, force=force
        )
        assert a_hat.shape == (bs, seq, 14), f"[{s}] a_hat shape wrong: {a_hat.shape}"
        l1 = torch.nn.functional.l1_loss(actions, a_hat)
        # 简单 KL
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        loss = l1 + 0.1 * kld

        # 门控辅助损失（接触真值取左右臂 OR 并集）
        alpha = getattr(model, "latest_alpha", None)
        if alpha is not None:
            contact_or = contact_gt.amax(dim=-1, keepdim=True)
            loss = loss + 0.1 * torch.nn.functional.binary_cross_entropy(alpha, contact_or)

        assert not torch.isnan(loss), f"[{s}] Loss NaN!"
        loss.backward()

        # 2. 推理（带 force）
        model.eval()
        with torch.no_grad():
            act_pred = model(qpos, image, None, force=force)
            assert act_pred[0].shape == (bs, seq, 14), f"[{s}] infer shape wrong"
            # 3. 退化（无 force）
            act_nf = model(qpos, image, None, force=None)
            assert act_nf[0].shape == (bs, seq, 14)

        print(f"  [OK] 方案 [{s:<13}] train_loss={loss.item():.4f}  "
              f"infer={tuple(act_pred[0].shape)}  alpha={tuple(alpha.shape) if alpha is not None else None}")

    print("\n全部 5 个方案本地验证通过！")


if __name__ == "__main__":
    run()
