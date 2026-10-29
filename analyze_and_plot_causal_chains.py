#!/usr/bin/env python3
"""跨实验组自动寻优 + "五层因果链"同步对比出图。

用法示例：
    python analyze_and_plot_causal_chains.py --task_name press_stapler
    python analyze_and_plot_causal_chains.py --task_name turn_switch --task_config demo_clean
    python analyze_and_plot_causal_chains.py --task_name press_stapler --seed 100014 100041

脚本读取
    eval_result/{task_name}/{policy}/{task_config}/{ckpt_setting}/<timestamp>/
        eval_metrics.json
        curves_per_episode/ep{episode_id:03d}_seed{seed}.npz

并完成三件事：
1. 自动筛选 Type A/B/C 三类典型对比种子并在终端输出推荐列表；
2. 绘制 5 行 1 列（共用时间步横轴）的论文级因果链对比图（PNG + PDF），
   以及"同模型成功轨迹 vs 失败轨迹"的自身对照图；
3. 输出包含 F_max > 1000 N 穿模奇异点剔除后的截尾均值与中位数的鲁棒统计表。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = ROOT / "eval_result"
DEFAULT_OUT_DIR = ROOT / "analysis"

EXPERIMENTS = (
    "exp01_concat_visual_only",
    "exp01_concat",
    "exp02_hard",
    "exp03_attn",
    "exp04_soft",
)
SOFT_EXP = "exp04_soft"
REVERSAL_BASELINES = ("exp01_concat_visual_only", "exp01_concat", "exp02_hard")
COMPLIANT_BASELINES = ("exp01_concat", "exp02_hard")

COLORS = {
    "exp01_concat_visual_only": "#8c8c8c",
    "exp01_concat": "#1f77b4",
    "exp02_hard": "#ff7f0e",
    "exp03_attn": "#9467bd",
    "exp04_soft": "#2ca02c",
}
LABELS = {
    "exp01_concat_visual_only": "Visual only (no force)",
    "exp01_concat": "Concat (hard fusion)",
    "exp02_hard": "Hard gate",
    "exp03_attn": "Cross-attention gate",
    "exp04_soft": "Soft gate (ours)",
}

CONTACT_THRESHOLD = 0.1
DANGER_FORCE_THRESHOLD = 10.0
OUTLIER_FORCE_THRESHOLD = 1000.0


# ============================ 数据读取 ============================


@dataclass
class Episode:
    experiment: str
    seed: int
    episode_id: int
    success: bool
    fail_type: str
    f_max: float
    force_var: float
    chunk_correction: float
    violation_rate: float
    raw: dict[str, Any] = field(default_factory=dict)
    curves: dict[str, np.ndarray] | None = None

    @property
    def has_curves(self) -> bool:
        return bool(self.curves)


def _latest_metrics_file(exp_dir: Path) -> Path | None:
    candidates = sorted(
        exp_dir.glob("**/eval_metrics.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _load_curves(curves_dir: Path, episode_id: int, seed: int) -> dict[str, np.ndarray] | None:
    path = curves_dir / f"ep{int(episode_id):03d}_seed{int(seed)}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:  # pragma: no cover - 损坏文件不应中断整轮分析
        print(f"[warn] failed to read {path}: {exc}")
        return None


def load_experiment(
    results_root: Path,
    task_name: str,
    policy: str,
    task_config: str,
    experiment: str,
) -> dict[int, Episode]:
    exp_dir = results_root / task_name / policy / task_config / experiment
    metrics_path = _latest_metrics_file(exp_dir) if exp_dir.is_dir() else None
    if metrics_path is None:
        print(f"[warn] no eval_metrics.json for {experiment} under {exp_dir}")
        return {}

    with metrics_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    curves_dir = metrics_path.parent / "curves_per_episode"

    episodes: dict[int, Episode] = {}
    for raw in payload.get("episodes", []):
        seed = int(raw.get("seed", -1))
        episode_id = int(raw.get("episode_id", -1))
        if seed < 0:
            continue
        episodes[seed] = Episode(
            experiment=experiment,
            seed=seed,
            episode_id=episode_id,
            success=bool(raw.get("success", False)),
            fail_type=str(raw.get("fail_type", "Unknown")),
            f_max=_as_float(raw.get("f_max")),
            force_var=_as_float(raw.get("force_var")),
            chunk_correction=_as_float(raw.get("chunk_correction")),
            violation_rate=_as_float(raw.get("violation_rate")),
            raw=raw,
            curves=_load_curves(curves_dir, episode_id, seed),
        )
    print(
        f"[load] {experiment:<26} runs={metrics_path.parent.name} "
        f"episodes={len(episodes)} curves={sum(1 for ep in episodes.values() if ep.has_curves)}"
    )
    return episodes


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


# ============================ 鲁棒统计 ============================


def robust_stats(values: Iterable[float], upper: float = OUTLIER_FORCE_THRESHOLD) -> dict[str, float]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "trimmed_mean": float("nan"), "median": float("nan"), "outliers": 0}
    kept = finite[finite <= upper]
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "trimmed_mean": float(kept.mean()) if kept.size else float("nan"),
        "median": float(np.median(finite)),
        "outliers": int(finite.size - kept.size),
    }


def print_robust_table(table: dict[str, dict[str, dict[str, float]]]) -> None:
    print("\n=== 力学安全鲁棒统计（剔除 F_max > 1000 N 的刚体穿模奇异点）===")
    header = (
        f"{'experiment':<26}{'n':>4}{'F_max mean±std':>26}"
        f"{'F_max trimmed':>15}{'F_max median':>14}{'outlier':>9}"
        f"{'F_var median':>14}"
    )
    print(header)
    print("-" * len(header))
    for experiment, stats in table.items():
        fmax = stats["f_max"]
        fvar = stats["force_var"]
        mean_std = (
            f"{fmax['mean']:.2f}±{fmax['std']:.2f}" if fmax["n"] else "N/A"
        )
        print(
            f"{experiment:<26}{int(fmax['n']):>4}{mean_std:>26}"
            f"{_fmt(fmax['trimmed_mean'], 15)}{_fmt(fmax['median'], 14)}"
            f"{int(fmax['outliers']):>9}{_fmt(fvar['median'], 14)}"
        )


def _fmt(value: float, width: int = 0, digits: int = 4) -> str:
    text = "N/A" if not np.isfinite(value) else f"{value:.{digits}f}"
    return f"{text:>{width}}" if width else text


# ============================ 典型种子自动筛选 ============================


@dataclass
class SeedCandidate:
    seed: int
    score: float
    detail: str


def _seeds(episodes: dict[int, Episode]) -> set[int]:
    return set(episodes)


def find_type_a(data: dict[str, dict[int, Episode]]) -> list[SeedCandidate]:
    soft = data.get(SOFT_EXP, {})
    baselines = {
        name: data.get(name, {})
        for name in REVERSAL_BASELINES
        if data.get(name)
    }
    candidates: list[SeedCandidate] = []
    for seed, ep in soft.items():
        if not ep.success or ep.f_max <= CONTACT_THRESHOLD or ep.chunk_correction <= 0:
            continue
        stuck = [
            name
            for name, episodes in baselines.items()
            if seed in episodes
            and not episodes[seed].success
            and episodes[seed].fail_type in ("Grasp_Success", "Missed")
        ]
        if not stuck:
            continue
        weight = 2.0 if "exp01_concat_visual_only" in stuck else 1.0
        weight += 1.0 if "exp02_hard" in stuck else 0.0
        candidates.append(
            SeedCandidate(
                seed=seed,
                score=weight * ep.chunk_correction,
                detail=(
                    f"soft=Task_Full_Success(f_max={ep.f_max:.2f}N, corr={ep.chunk_correction:.3f}) "
                    f"stuck_in={','.join(stuck)}"
                ),
            )
        )
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def find_type_b(data: dict[str, dict[int, Episode]]) -> list[SeedCandidate]:
    soft = data.get(SOFT_EXP, {})
    candidates: list[SeedCandidate] = []
    for seed, ep in soft.items():
        if not ep.success:
            continue
        ratios: list[float] = []
        detail_parts: list[str] = []
        for name in COMPLIANT_BASELINES:
            baseline = data.get(name, {}).get(seed)
            if baseline is None:
                continue
            detail_parts.append(f"{name}=f{baseline.f_max:.2f}N/var{baseline.force_var:.3f}")
            if baseline.f_max > 1e-6:
                ratios.append(ep.f_max / baseline.f_max)
        if not ratios:
            continue
        ratio = min(ratios)
        if ratio >= 1.0:
            continue
        candidates.append(
            SeedCandidate(
                seed=seed,
                score=1.0 / max(ratio, 1e-6),
                detail=(
                    f"soft=f{ep.f_max:.2f}N/var{ep.force_var:.3f} "
                    f"vs " + ", ".join(detail_parts) + f" (ratio={ratio:.2f})"
                ),
            )
        )
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def find_type_c(data: dict[str, dict[int, Episode]]) -> list[SeedCandidate]:
    soft = data.get(SOFT_EXP, {})
    candidates: list[SeedCandidate] = []
    for seed, ep in soft.items():
        if ep.success:
            continue
        for name in REVERSAL_BASELINES:
            baseline = data.get(name, {}).get(seed)
            if baseline is None or baseline.success:
                continue
            if ep.f_max <= 0 or baseline.f_max <= DANGER_FORCE_THRESHOLD:
                continue
            ratio = baseline.f_max / ep.f_max
            if ratio <= 1.0:
                continue
            candidates.append(
                SeedCandidate(
                    seed=seed,
                    score=ratio,
                    detail=(
                        f"{name}=f_max {baseline.f_max:.2f}N vs soft={ep.f_max:.2f}N "
                        f"(x{ratio:.2f} lower, both {baseline.fail_type}/{ep.fail_type})"
                    ),
                )
            )
            break
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def print_candidates(kind: str, title: str, candidates: list[SeedCandidate], limit: int = 5) -> None:
    print(f"\n--- Type {kind}: {title} ---")
    if not candidates:
        print("  (no matching seed)")
        return
    for rank, candidate in enumerate(candidates[:limit], start=1):
        print(f"  {rank}. seed={candidate.seed}  {candidate.detail}")


# ============================ 绘图 ============================


def _experiment_curve(
    episodes: dict[int, Episode], seed: int, key: str
) -> np.ndarray | None:
    ep = episodes.get(seed)
    if ep is None or not ep.curves:
        return None
    array = ep.curves.get(key)
    return None if array is None else np.asarray(array).reshape(array.shape[0], -1).squeeze()


def _contact_spans(contact: np.ndarray | None) -> list[tuple[int, int]]:
    if contact is None or contact.size == 0:
        return []
    mask = np.asarray(contact).reshape(-1) > 0.5
    spans: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(mask) - 1))
    return spans


def _first_success_step(step_success: np.ndarray | None) -> int | None:
    if step_success is None or step_success.size == 0:
        return None
    hits = np.nonzero(np.asarray(step_success).reshape(-1) > 0.5)[0]
    return int(hits[0]) if hits.size else None


def _progress_axis(progress: np.ndarray | None) -> tuple[int, str]:
    """末端按压主轴：位移幅度最大的坐标轴。"""
    if progress is None:
        return 2, "z"
    finite = np.asarray(progress, dtype=np.float64)
    if finite.ndim == 1:
        finite = finite.reshape(-1, 1)
    if finite.size == 0 or np.all(np.isnan(finite)):
        return 2, "z"
    ranges = np.nanmax(finite, axis=0) - np.nanmin(finite, axis=0)
    axis = int(np.nanargmax(ranges))
    return axis, "xyz"[axis] if axis < 3 else f"a{axis}"


def plot_five_layer(
    seed: int,
    data: dict[str, dict[int, Episode]],
    experiments: list[str],
    out_png: Path,
    title_prefix: str = "",
) -> Path | None:
    plotted = [name for name in experiments if _experiment_curve(data.get(name, {}), seed, "force_norm") is not None]
    if not plotted:
        print(f"[warn] seed={seed} has no curves to plot")
        return None

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    ax_force, ax_alpha, ax_corr, ax_tcp, ax_prog = axes

    reference = SOFT_EXP if SOFT_EXP in plotted else plotted[0]
    reference_contact = _experiment_curve(data.get(reference, {}), seed, "contact_gt")
    for start, end in _contact_spans(reference_contact):
        for axis in axes:
            axis.axvspan(start, end, color="#4c72b0", alpha=0.10, lw=0)
    ax_force.axhline(DANGER_FORCE_THRESHOLD, color="#d62728", ls="--", lw=1.0)
    ax_force.text(
        0.005, DANGER_FORCE_THRESHOLD, " 10 N safety limit",
        transform=ax_force.get_yaxis_transform(), color="#d62728",
        fontsize=8, va="bottom",
    )

    summary_lines: list[str] = []
    for name in plotted:
        episodes = data.get(name, {})
        color = COLORS.get(name, None)
        label = LABELS.get(name, name)
        ep = episodes.get(seed)
        suffix = "" if ep is None else f" [{'OK' if ep.success else ep.fail_type}]"
        kwargs = dict(color=color, lw=1.4, label=label + suffix)

        force = _experiment_curve(episodes, seed, "force_norm")
        if force is not None:
            ax_force.plot(np.arange(force.size), force, **kwargs)
        alpha = _experiment_curve(episodes, seed, "alpha_gate")
        if alpha is not None:
            ax_alpha.plot(np.arange(alpha.size), alpha, **kwargs)
        corr = _experiment_curve(episodes, seed, "step_chunk_correction")
        if corr is not None:
            ax_corr.plot(np.arange(corr.size), corr, **kwargs)
        tcp = _experiment_curve(episodes, seed, "tcp_pos")
        if tcp is not None and np.any(np.isfinite(tcp)):
            axis_index, axis_name = _progress_axis(tcp)
            ax_tcp.plot(np.arange(tcp.shape[0]), tcp[:, axis_index], **kwargs)
            ax_tcp.set_ylabel(f"TCP {axis_name} (m)")
        progress = _experiment_curve(episodes, seed, "obj_progress_norm")
        if progress is None or not np.any(np.isfinite(progress)):
            progress = _experiment_curve(episodes, seed, "obj_progress")
        if progress is not None:
            ax_prog.plot(np.arange(progress.size), progress, **kwargs)
        success_step = _first_success_step(_experiment_curve(episodes, seed, "step_success"))
        if success_step is not None:
            ax_prog.axvline(success_step, color=color, ls=":", lw=1.2)
            summary_lines.append(f"{label}: Task_Full_Success @ t={success_step}")

    ax_force.set_ylabel(r"$\|f_t\|_2$ (N)")
    ax_alpha.set_ylabel(r"$\alpha_{gate}(t)$")
    ax_corr.set_ylabel("chunk correction\n(rel. L2)")
    if ax_tcp.get_ylabel() == "":
        ax_tcp.set_ylabel("TCP (m)")
    ax_prog.set_ylabel("object progress\n(normalized)")
    ax_prog.set_xlabel("time step $t$")

    ax_alpha.set_ylim(-0.05, 1.05)
    ax_force.set_yscale("symlog", linthresh=1.0)
    ax_force.legend(fontsize=8, loc="upper left")
    ax_prog.legend(fontsize=8, loc="lower right")

    for axis, tag in zip(
        axes, ("1. contact force", "2. gate weight", "3. action correction",
               "4. end-effector response", "5. object progress"),
    ):
        axis.grid(alpha=0.25, lw=0.6)
        axis.set_title(tag, loc="left", fontsize=9, color="#444444")

    fig.suptitle(
        f"{title_prefix}seed={seed} causal chain (contact -> gate -> action -> motion -> success)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[plot] {out_png} (+ .pdf)")
    for line in summary_lines:
        print(f"        {line}")
    return out_png


def plot_self_comparison(
    experiment: str,
    success_ep: Episode,
    failure_ep: Episode,
    out_png: Path,
) -> Path | None:
    if not success_ep.has_curves or not failure_ep.has_curves:
        print(f"[warn] missing curves for self-comparison of {experiment}")
        return None

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)
    ax_force, ax_alpha, ax_corr, ax_tcp, ax_prog = axes
    pairs = (
        (success_ep, "#2ca02c", "success", "-"),
        (failure_ep, "#d62728", f"{failure_ep.fail_type}", "--"),
    )

    for ep, color, tag, style in pairs:
        curves = ep.curves or {}
        label = f"seed={ep.seed} {tag}"
        steps = int(np.asarray(curves.get("time_step", np.arange(0))).size)
        x = np.arange(steps)
        for key, axis in (
            ("force_norm", ax_force),
            ("alpha_gate", ax_alpha),
            ("step_chunk_correction", ax_corr),
        ):
            values = curves.get(key)
            if values is not None:
                axis.plot(
                    x[: len(values)], np.asarray(values),
                    color=color, lw=1.4, ls=style, label=label,
                )
        tcp = curves.get("tcp_pos")
        if tcp is not None:
            axis_index, _ = _progress_axis(tcp)
            ax_tcp.plot(
                x[: len(tcp)], np.asarray(tcp)[:, axis_index],
                color=color, lw=1.4, ls=style, label=label,
            )
        progress = curves.get("obj_progress_norm", curves.get("obj_progress"))
        if progress is not None:
            values = np.asarray(progress)
            ax_prog.plot(
                x[: len(values)], values, color=color, lw=1.4, ls=style, label=label
            )
        success_step = _first_success_step(curves.get("step_success"))
        if success_step is not None:
            ax_prog.axvline(success_step, color=color, ls=":", lw=1.2)
            ax_prog.annotate(
                "Task_Full_Success",
                xy=(success_step, 0.95),
                xycoords=("data", "axes fraction"),
                fontsize=8,
                color=color,
                ha="left",
            )
        contact = curves.get("contact_gt")
        for start, end in _contact_spans(contact):
            ax_force.axvspan(start, end, color=color, alpha=0.08, lw=0)

    ax_force.axhline(DANGER_FORCE_THRESHOLD, color="#d62728", ls="--", lw=1.0)
    ax_force.set_yscale("symlog", linthresh=1.0)
    ax_force.set_ylabel(r"$\|f_t\|_2$ (N)")
    ax_alpha.set_ylabel(r"$\alpha_{gate}(t)$")
    ax_alpha.set_ylim(-0.05, 1.05)
    ax_corr.set_ylabel("chunk correction")
    ax_tcp.set_ylabel("TCP (m)")
    ax_prog.set_ylabel("object progress")
    ax_prog.set_xlabel("time step $t$")
    ax_force.legend(fontsize=8)
    for axis, tag in zip(
        axes, ("1. contact force", "2. gate weight", "3. action correction",
               "4. end-effector response", "5. object progress"),
    ):
        axis.grid(alpha=0.25, lw=0.6)
        axis.set_title(tag, loc="left", fontsize=9, color="#444444")
    fig.suptitle(
        f"{experiment}: successful seed={success_ep.seed} vs "
        f"{failure_ep.fail_type} seed={failure_ep.seed}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[plot] {out_png} (+ .pdf)")
    return out_png


# ============================ 主流程 ============================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动筛选典型种子并绘制五层因果链对比图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task_name", required=True, help="任务名，例如 press_stapler")
    parser.add_argument("--task_config", default="demo_clean", help="任务配置名")
    parser.add_argument("--policy", default="ACT", help="策略目录名")
    parser.add_argument("--results_root", default=str(DEFAULT_RESULTS_ROOT), help="eval_result 根目录")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="图表输出根目录")
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), help="实验组目录名列表")
    parser.add_argument("--seed", nargs="+", type=int, default=None, help="指定种子（默认自动取 Top-K）")
    parser.add_argument("--top_k", type=int, default=3, help="自动筛选每种类型取前 K 个种子出图")
    parser.add_argument("--list_only", action="store_true", help="只打印推荐种子，不出图")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve() / args.task_name / args.task_config

    print(f"task={args.task_name} config={args.task_config} policy={args.policy}")
    print(f"results_root={results_root}")
    data = {
        name: load_experiment(results_root, args.task_name, args.policy, args.task_config, name)
        for name in args.experiments
    }

    table = {
        name: {
            "f_max": robust_stats(ep.f_max for ep in episodes.values()),
            "force_var": robust_stats(ep.force_var for ep in episodes.values()),
        }
        for name, episodes in data.items()
        if episodes
    }
    if not table:
        raise SystemExit("no eval_metrics.json found; run the evaluation first")
    print_robust_table(table)

    type_a = find_type_a(data)
    type_b = find_type_b(data)
    type_c = find_type_c(data)
    print_candidates("A", "因果逆转成功种子 (baseline stuck / soft succeeds)", type_a)
    print_candidates("B", "同成功但柔顺降力种子 (soft lowers force)", type_b)
    print_candidates("C", "典型受阻保护种子 (baseline overload / soft protects)", type_c)

    if args.list_only:
        return

    if args.seed:
        chosen = list(dict.fromkeys(args.seed))
        print(f"\n[plot] using user-specified seeds: {chosen}")
    else:
        picked: list[int] = []
        for candidates in (type_a, type_b, type_c):
            for candidate in candidates[: args.top_k]:
                if candidate.seed not in picked:
                    picked.append(candidate.seed)
        chosen = picked[: max(args.top_k, 3)]

    for seed in chosen:
        plot_five_layer(seed, data, list(args.experiments), out_root / f"causal_chain_seed{seed}.png")

    for experiment, episodes in data.items():
        if not episodes:
            continue
        successes = [ep for ep in episodes.values() if ep.success and ep.has_curves]
        failures = [
            ep for ep in episodes.values()
            if not ep.success and ep.has_curves and ep.fail_type == "Grasp_Success"
        ]
        if not failures:
            failures = [ep for ep in episodes.values() if not ep.success and ep.has_curves]
        if not successes or not failures:
            continue
        best_failure = max(failures, key=lambda ep: ep.f_max)
        plot_self_comparison(
            experiment,
            successes[0],
            best_failure,
            out_root / f"self_compare_{experiment}_seed{successes[0].seed}_vs_{best_failure.seed}.png",
        )

    print(f"\nAll figures written under {out_root}")


if __name__ == "__main__":
    main()
