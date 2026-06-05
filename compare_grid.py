#!/usr/bin/env python3
"""
Full Suite 2 grid comparison: policies × agent/obstacle configs × noise.

Dynamics fixed to dynamic. Parses evaluation summaries under results/ and
writes figures to results/plots/grid/ by default (60 experiments).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from compare_dynamics import (
    FAILURE_COLORS,
    FAILURE_KEYS,
    POLICY_LABELS,
    TRAIN_TYPES,
    EvalResult,
    parse_result_file,
)

plt.rcParams["font.family"] = "serif"

DYNAMICS = "dynamic"
NOISE_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
AGENT_OBSTACLE_CONFIGS: Tuple[Tuple[int, int], ...] = (
    (5, 0),
    (4, 1),
    (3, 2),
    (2, 3),
    (1, 4),
)

NOISE_LABELS = {"low": "Low", "medium": "Medium", "high": "High"}
NOISE_COLORS = {"low": "#27ae60", "medium": "#f39c12", "high": "#c0392b"}

POLICY_COLORS = {
    "naive": "#95a5a6",
    "cbf": "#2ecc71",
    "reward_only": "#9b59b6",
    "filter_only": "#e67e22",
}

GridKey = Tuple[str, int, int, str]  # policy, agents, obstacles, noise
ConditionKey = Tuple[int, int, str]  # agents, obstacles, noise


def config_label(agents: int, obstacles: int) -> str:
    return f"{agents}A/{obstacles}O"


def condition_label(agents: int, obstacles: int, noise: str) -> str:
    return f"{config_label(agents, obstacles)} · {NOISE_LABELS[noise]}"


def iter_conditions() -> List[ConditionKey]:
    out: List[ConditionKey] = []
    for agents, obstacles in AGENT_OBSTACLE_CONFIGS:
        for noise in NOISE_LEVELS:
            out.append((agents, obstacles, noise))
    return out


def discover_results(
    results_root: Path,
    dynamics: str = DYNAMICS,
) -> Dict[GridKey, EvalResult]:
    allowed_configs = set(AGENT_OBSTACLE_CONFIGS)
    allowed_noise = set(NOISE_LEVELS)
    candidates: Dict[GridKey, List[EvalResult]] = {}

    for path in sorted(results_root.glob("results_*/*/*.txt")):
        parsed = parse_result_file(path)
        if parsed is None:
            continue
        if parsed.dynamics != dynamics:
            continue
        if parsed.policy not in TRAIN_TYPES:
            continue
        if (parsed.num_agents, parsed.num_obstacles) not in allowed_configs:
            continue
        if parsed.control_noise not in allowed_noise:
            continue

        key: GridKey = (
            parsed.policy,
            parsed.num_agents,
            parsed.num_obstacles,
            parsed.control_noise,
        )
        candidates.setdefault(key, []).append(parsed)

    selected: Dict[GridKey, EvalResult] = {}
    for key, items in candidates.items():
        items.sort(key=lambda r: r.source_file)
        if len(items) > 1:
            print(
                f"Warning: multiple matches for {key}; using {items[-1].source_file}"
            )
        selected[key] = items[-1]
    return selected


def require_complete(results: Dict[GridKey, EvalResult]) -> None:
    missing = [
        (p, a, o, n)
        for p in TRAIN_TYPES
        for a, o in AGENT_OBSTACLE_CONFIGS
        for n in NOISE_LEVELS
        if (p, a, o, n) not in results
    ]
    if missing:
        lines = [
            f"  - {POLICY_LABELS[p]} / {condition_label(a, o, n)}"
            for p, a, o, n in missing[:20]
        ]
        extra = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise SystemExit(
            f"Missing grid results ({DYNAMICS}): {len(missing)} cells\n"
            + "\n".join(lines)
            + extra
        )


def get_result(
    results: Dict[GridKey, EvalResult],
    policy: str,
    agents: int,
    obstacles: int,
    noise: str,
) -> EvalResult:
    return results[(policy, agents, obstacles, noise)]


def overview_matrix(
    results: Dict[GridKey, EvalResult],
    value_fn,
) -> Tuple[np.ndarray, List[str]]:
    """Rows = conditions, cols = policies."""
    conditions = iter_conditions()
    mat = np.zeros((len(conditions), len(TRAIN_TYPES)))
    labels = [condition_label(a, o, n) for a, o, n in conditions]
    for i, (a, o, n) in enumerate(conditions):
        for j, policy in enumerate(TRAIN_TYPES):
            mat[i, j] = value_fn(get_result(results, policy, a, o, n))
    return mat, labels


def plot_overview_heatmap(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    mat, row_labels = overview_matrix(
        results, lambda r: r.success_rate * 100
    )
    fig_h = max(8, 0.35 * len(row_labels))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(TRAIN_TYPES)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f"Success rate (%) — full grid ({DYNAMICS})")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color = "white" if val < 45 or val > 85 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=7)

    # Separate config groups
    for k in range(1, len(AGENT_OBSTACLE_CONFIGS)):
        ax.axhline(k * len(NOISE_LEVELS) - 0.5, color="white", linewidth=2)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Success (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_policy_facets(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """2×2 heatmaps: each policy, rows=config, cols=noise."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, policy in zip(axes.flat, TRAIN_TYPES):
        mat = np.zeros((len(AGENT_OBSTACLE_CONFIGS), len(NOISE_LEVELS)))
        for i, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
            for j, noise in enumerate(NOISE_LEVELS):
                mat[i, j] = get_result(results, policy, a, o, noise).success_rate * 100

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_title(POLICY_LABELS[policy], fontsize=11)
        ax.set_xticks(range(len(NOISE_LEVELS)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
        ax.set_yticks(range(len(AGENT_OBSTACLE_CONFIGS)))
        ax.set_yticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS], fontsize=8)
        ax.set_xlabel("Noise")
        ax.set_ylabel("Agents / obstacles")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val < 45 or val > 85 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Success rate by policy ({DYNAMICS})", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_per_noise(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """1×3: config × policy for each noise level."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for ax, noise in zip(axes, NOISE_LEVELS):
        mat = np.zeros((len(AGENT_OBSTACLE_CONFIGS), len(TRAIN_TYPES)))
        for i, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
            for j, policy in enumerate(TRAIN_TYPES):
                mat[i, j] = get_result(results, policy, a, o, noise).success_rate * 100

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_title(f"{NOISE_LABELS[noise]} noise")
        ax.set_xticks(range(len(TRAIN_TYPES)))
        ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES], rotation=20, ha="right")
        ax.set_yticks(range(len(AGENT_OBSTACLE_CONFIGS)))
        ax.set_yticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
        ax.set_xlabel("Policy")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if val < 45 or val > 85 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Agents / obstacles")
    fig.suptitle(f"Success rate by noise level ({DYNAMICS})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lines_by_noise(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """For each noise: success vs config, one line per policy."""
    x = np.arange(len(AGENT_OBSTACLE_CONFIGS))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, noise in zip(axes, NOISE_LEVELS):
        for policy in TRAIN_TYPES:
            vals = [
                get_result(results, policy, a, o, noise).success_rate * 100
                for a, o in AGENT_OBSTACLE_CONFIGS
            ]
            ax.plot(
                x, vals, marker="o", linewidth=2, label=POLICY_LABELS[policy],
                color=POLICY_COLORS[policy],
            )
        ax.set_title(f"{NOISE_LABELS[noise]} noise")
        ax.set_xticks(x)
        ax.set_xticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS], fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 105)

    axes[0].set_ylabel("Success rate (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.set_xlabel("Agents / obstacles")
    fig.suptitle(f"Success vs density at each noise level ({DYNAMICS})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lines_by_config(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """For each config: success vs noise, one line per policy."""
    x = np.arange(len(NOISE_LEVELS))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes_flat = list(axes.flat)

    for idx, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
        ax = axes_flat[idx]
        for policy in TRAIN_TYPES:
            vals = [
                get_result(results, policy, a, o, n).success_rate * 100
                for n in NOISE_LEVELS
            ]
            ax.plot(
                x, vals, marker="o", linewidth=2, label=POLICY_LABELS[policy],
                color=POLICY_COLORS[policy],
            )
        ax.set_title(config_label(a, o))
        ax.set_xticks(x)
        ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, loc="center", fontsize=10, frameon=False)

    for ax in axes_flat[:5]:
        ax.set_xlabel("Noise")
    axes_flat[0].set_ylabel("Success rate (%)")
    axes_flat[3].set_ylabel("Success rate (%)")

    fig.suptitle(f"Success vs noise at each density ({DYNAMICS})", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_policy_map(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """Categorical map: which policy wins at each (config, noise) cell."""
    mat = np.zeros((len(AGENT_OBSTACLE_CONFIGS), len(NOISE_LEVELS)), dtype=int)
    for i, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
        for j, noise in enumerate(NOISE_LEVELS):
            rates = [
                get_result(results, p, a, o, noise).success_rate
                for p in TRAIN_TYPES
            ]
            mat[i, j] = int(np.argmax(rates))

    cmap = plt.cm.colors.ListedColormap(
        [POLICY_COLORS[p] for p in TRAIN_TYPES]
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=len(TRAIN_TYPES) - 1, aspect="auto")

    ax.set_xticks(range(len(NOISE_LEVELS)))
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
    ax.set_yticks(range(len(AGENT_OBSTACLE_CONFIGS)))
    ax.set_yticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
    ax.set_xlabel("Control noise")
    ax.set_ylabel("Agents / obstacles")
    ax.set_title(f"Best policy by success rate ({DYNAMICS})")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            policy = TRAIN_TYPES[mat[i, j]]
            rate = get_result(results, policy, *AGENT_OBSTACLE_CONFIGS[i], NOISE_LEVELS[j])
            ax.text(
                j, i, f"{POLICY_LABELS[policy][:3]}\n{rate.success_rate * 100:.0f}%",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold",
            )

    patches = [
        mpatches.Patch(color=POLICY_COLORS[p], label=POLICY_LABELS[p])
        for p in TRAIN_TYPES
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_margin_over_naive(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """3 heatmaps: CBF/Reward/Filter success minus naive (pp)."""
    alt_policies = ("cbf", "reward_only", "filter_only")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    vmax = 0.0

    mats = []
    for alt in alt_policies:
        mat = np.zeros((len(AGENT_OBSTACLE_CONFIGS), len(NOISE_LEVELS)))
        for i, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
            for j, noise in enumerate(NOISE_LEVELS):
                naive_sr = get_result(results, "naive", a, o, noise).success_rate
                alt_sr = get_result(results, alt, a, o, noise).success_rate
                mat[i, j] = (alt_sr - naive_sr) * 100
        mats.append(mat)
        vmax = max(vmax, np.max(np.abs(mat)))

    vmax = max(vmax, 5)
    for ax, alt, mat in zip(axes, alt_policies, mats):
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{POLICY_LABELS[alt]} − Naive")
        ax.set_xticks(range(len(NOISE_LEVELS)))
        ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
        ax.set_yticks(range(len(AGENT_OBSTACLE_CONFIGS)))
        ax.set_yticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
        ax.set_xlabel("Noise")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(j, i, f"{val:+.0f}", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pp")

    axes[0].set_ylabel("Agents / obstacles")
    fig.suptitle(f"Success margin over naive ({DYNAMICS})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_policy_summary_bars(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """Mean and std of success over all 15 conditions per policy."""
    means, stds = [], []
    for policy in TRAIN_TYPES:
        vals = [
            get_result(results, policy, a, o, n).success_rate * 100
            for a, o in AGENT_OBSTACLE_CONFIGS
            for n in NOISE_LEVELS
        ]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(TRAIN_TYPES))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5,
        color=[POLICY_COLORS[p] for p in TRAIN_TYPES],
        edgecolor="black", linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_ylabel("Success rate (%)")
    ax.set_title(f"Mean success ± std over 15 conditions ({DYNAMICS})")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + s + 2,
            f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_overview(
    results: Dict[GridKey, EvalResult],
    out_path: Path,
    failure_key: str,
) -> None:
    mat, row_labels = overview_matrix(
        results, lambda r: r.failure_rates[failure_key] * 100
    )
    fig_h = max(8, 0.35 * len(row_labels))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    vmax = max(15, mat.max() * 1.05)
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(TRAIN_TYPES)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(f"{failure_key.capitalize()} failure rate (%) — full grid ({DYNAMICS})")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val < 0.05:
                continue
            color = "white" if val > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=7)

    for k in range(1, len(AGENT_OBSTACLE_CONFIGS)):
        ax.axhline(k * len(NOISE_LEVELS) - 0.5, color="white", linewidth=2)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_noise_sensitivity(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """Per policy: mean |Δsuccess| between adjacent noise levels, averaged over configs."""
    x = np.arange(len(TRAIN_TYPES))
    sensitivities = []

    for policy in TRAIN_TYPES:
        deltas = []
        for a, o in AGENT_OBSTACLE_CONFIGS:
            rates = [
                get_result(results, policy, a, o, n).success_rate * 100
                for n in NOISE_LEVELS
            ]
            deltas.append(abs(rates[1] - rates[0]) + abs(rates[2] - rates[1]))
        sensitivities.append(np.mean(deltas) / 2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        x, sensitivities,
        color=[POLICY_COLORS[p] for p in TRAIN_TYPES],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_ylabel("Mean |Δsuccess| between noise levels (pp)")
    ax.set_title(f"Noise sensitivity by policy ({DYNAMICS})")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, sensitivities):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.15,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_density_sensitivity(
    results: Dict[GridKey, EvalResult], out_path: Path
) -> None:
    """Per policy: mean |Δsuccess| between adjacent configs, averaged over noise."""
    x = np.arange(len(TRAIN_TYPES))
    sensitivities = []

    for policy in TRAIN_TYPES:
        deltas = []
        for noise in NOISE_LEVELS:
            rates = [
                get_result(results, policy, a, o, noise).success_rate * 100
                for a, o in AGENT_OBSTACLE_CONFIGS
            ]
            step_deltas = [abs(rates[i + 1] - rates[i]) for i in range(len(rates) - 1)]
            deltas.append(np.mean(step_deltas))
        sensitivities.append(np.mean(deltas))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        x, sensitivities,
        color=[POLICY_COLORS[p] for p in TRAIN_TYPES],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_ylabel("Mean |Δsuccess| between density steps (pp)")
    ax.set_title(f"Agent/obstacle density sensitivity ({DYNAMICS})")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, sensitivities):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(results: Dict[GridKey, EvalResult], out_path: Path) -> None:
    header = (
        "policy,dynamics,agents,obstacles,control_noise,episodes,"
        "success_rate_pct,mean_reward,failures_total,"
        "fail_obstacle,fail_wall,fail_timeout,training_run,source_file\n"
    )
    rows = []
    for policy in TRAIN_TYPES:
        for a, o in AGENT_OBSTACLE_CONFIGS:
            for noise in NOISE_LEVELS:
                r = get_result(results, policy, a, o, noise)
                rows.append(
                    f"{policy},{r.dynamics},{r.num_agents},{r.num_obstacles},{r.control_noise},"
                    f"{r.episodes},{r.success_rate * 100:.2f},{r.mean_reward:.2f},"
                    f"{r.failures_total},{r.failure_counts['obstacle']},"
                    f"{r.failure_counts['wall']},{r.failure_counts['timeout']},"
                    f"{r.training_run},{r.source_file}\n"
                )
    out_path.write_text(header + "".join(rows))


def print_summary(results: Dict[GridKey, EvalResult]) -> None:
    print(f"\nFull grid ({DYNAMICS}): {len(TRAIN_TYPES) * len(AGENT_OBSTACLE_CONFIGS) * len(NOISE_LEVELS)} experiments\n")

    for policy in TRAIN_TYPES:
        vals = [
            get_result(results, policy, a, o, n).success_rate * 100
            for a, o in AGENT_OBSTACLE_CONFIGS
            for n in NOISE_LEVELS
        ]
        print(
            f"  {POLICY_LABELS[policy]:<14} "
            f"mean={np.mean(vals):5.1f}%  min={np.min(vals):5.1f}%  max={np.max(vals):5.1f}%"
        )

    wins = {p: 0 for p in TRAIN_TYPES}
    for a, o in AGENT_OBSTACLE_CONFIGS:
        for noise in NOISE_LEVELS:
            best = max(
                TRAIN_TYPES,
                key=lambda p: get_result(results, p, a, o, noise).success_rate,
            )
            wins[best] += 1
    print("\nBest-policy wins (out of 15 conditions):")
    for p in TRAIN_TYPES:
        print(f"  {POLICY_LABELS[p]:<14} {wins[p]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot full Suite 2 grid (policies × density × noise, dynamic)."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/plots/grid"))
    parser.add_argument("--dynamics", type=str, default=DYNAMICS)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = discover_results(results_root, dynamics=args.dynamics)
    require_complete(results)
    print_summary(results)

    plots = {
        "overview_success_heatmap.png": plot_overview_heatmap,
        "policy_facets_heatmap.png": plot_policy_facets,
        "heatmap_per_noise.png": plot_heatmap_per_noise,
        "success_lines_by_noise.png": plot_lines_by_noise,
        "success_lines_by_config.png": plot_lines_by_config,
        "best_policy_map.png": plot_best_policy_map,
        "margin_over_naive.png": plot_margin_over_naive,
        "policy_mean_summary.png": plot_policy_summary_bars,
        "noise_sensitivity.png": plot_noise_sensitivity,
        "density_sensitivity.png": plot_density_sensitivity,
        "obstacle_failure_overview.png": lambda r, p: plot_failure_overview(r, p, "obstacle"),
        "timeout_failure_overview.png": lambda r, p: plot_failure_overview(r, p, "timeout"),
    }

    for filename, fn in plots.items():
        path = out_dir / filename
        fn(results, path)
        print(f"Saved {path}")

    csv_path = out_dir / "grid_summary.csv"
    write_summary_csv(results, csv_path)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
