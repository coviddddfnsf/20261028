#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eval_result" / "turn_switch" / "ACT" / "demo_clean"
EXPERIMENTS = (
    "exp01_concat_visual_only",
    "exp01_concat",
    "exp02_hard",
    "exp03_attn",
    "exp04_soft",
)


def latest_metrics(folder: str) -> dict:
    files = sorted(
        (RESULTS / folder).glob("**/eval_metrics.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No eval_metrics.json found for {folder}")
    with files[-1].open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    if not isinstance(metrics, dict) or not isinstance(metrics.get("summary"), dict):
        raise ValueError(f"Unexpected metrics JSON structure: {files[-1]}")
    return metrics["summary"]


def number(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def main() -> None:
    print(
        "| 实验组别 (Folder ID) | 标称成功率 SR (%) | Missed / Grasp / Success "
        "| 峰值冲击力 F_max (N) | 危险力超限率 Violation_Rate (%) "
        "| 动作抖动 Jerk (m/s3) | 动作修正量 Chunk_Corr |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for folder in EXPERIMENTS:
        summary = latest_metrics(folder)
        failures = summary.get("Fail_Type", {})
        counts = (
            failures.get("Missed", 0),
            failures.get("Grasp_Success", 0),
            failures.get("Task_Full_Success", 0),
        )
        row = (
            folder,
            number(summary.get("SR")),
            " / ".join(str(count) for count in counts),
            number((summary.get("F_max") or {}).get("mean")),
            number((summary.get("Violation_Rate") or {}).get("mean")),
            number((summary.get("Kinematic_Jerk") or {}).get("mean")),
            number((summary.get("Chunk_Correction") or {}).get("mean"), 4),
        )
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
