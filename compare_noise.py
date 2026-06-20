#!/usr/bin/env python3
"""
Compare control noise levels (Suite 2) across the four policy types.

Parses evaluation summary .txt files under results/ and generates plots under
results/plots/noise/ by default.

Fixed eval config: 1 agent, 4 obstacles, dynamic dynamics model.
Noise levels: low, medium, high.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

NOISE_LEVELS: Tuple[str, ...] = ("low", "medium", "high")
NUM_AGENTS = 1
NUM_OBSTACLES = 4
DYNAMICS = "dynamic"

NOISE_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}

NOISE_COLORS = {
    "low": "#27ae60",
    "medium": "#f39c12",
    "high": "#c0392b",
}

POLICY_COLORS = {
    "naive": "#95a5a6",
    "cbf": "#2ecc71",
    "reward_only": "#9b59b6",
    "filter_only": "#e67e22",
}


def discover_results(
    results_root: Path,
    dynamics: str = DYNAMICS,
    num_agents: int = NUM_AGENTS,
    num_obstacles: int = NUM_OBSTACLES,
) -> Dict[Tuple[str, str], EvalResult]:
    """One result per (policy, noise), preferring the newest file."""
    allowed_noise = set(NOISE_LEVELS)
    candidates: Dict[Tuple[str, str], List[EvalResult]] = {}

    for path in sorted(results_root.glob("results_*/*/*.txt")):
        parsed = parse_result_file(path)
        if parsed is None:
            continue
        if parsed.dynamics != dynamics:
            continue
        if parsed.num_agents != num_agents:
            continue
        if parsed.num_obstacles != num_obstacles:
            continue
        if parsed.control_noise not in allowed_noise:
            continue
        if parsed.policy not in TRAIN_TYPES:
            continue

        key = (parsed.policy, parsed.control_noise)
        candidates.setdefault(key, []).append(parsed)

    selected: Dict[Tuple[str, str], EvalResult] = {}
    for key, items in candidates.items():
        items.sort(key=lambda r: r.source_file)
        if len(items) > 1:
            print(
                f"Warning: multiple matches for {key[0]}/{key[1]}; "
                f"using {items[-1].source_file}"
            )
        selected[key] = items[-1]
    return selected


def require_complete(results: Dict[Tuple[str, str], EvalResult]) -> None:
    missing = [
        (policy, noise)
        for policy in TRAIN_TYPES
        for noise in NOISE_LEVELS
        if (policy, noise) not in results
    ]
    if missing:
        lines = [f"  - {POLICY_LABELS[p]} / {NOISE_LABELS[n]}" for p, n in missing]
        raise SystemExit(
            f"Missing noise comparison results "
            f"({NUM_AGENTS} agent, {NUM_OBSTACLES} obstacles, {DYNAMICS}):\n"
            + "\n".join(lines)
        )


def success_matrix(results: Dict[Tuple[str, str], EvalResult]) -> np.ndarray:
    """Rows = noise levels, cols = policies."""
    mat = np.zeros((len(NOISE_LEVELS), len(TRAIN_TYPES)))
    for i, noise in enumerate(NOISE_LEVELS):
        for j, policy in enumerate(TRAIN_TYPES):
            mat[i, j] = results[(policy, noise)].success_rate * 100.0
    return mat


def plot_success_heatmap(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    mat = success_matrix(results)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(TRAIN_TYPES)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_yticks(range(len(NOISE_LEVELS)))
    ax.set_yticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
    ax.set_xlabel("Training type")
    ax.set_ylabel("Control noise")
    ax.set_title(
        f"Success rate (%) — {NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS}"
    )

    for i in range(len(NOISE_LEVELS)):
        for j in range(len(TRAIN_TYPES)):
            val = mat[i, j]
            color = "white" if val < 45 or val > 85 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_bars(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    x = np.arange(len(TRAIN_TYPES))
    width = 0.25
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, noise in enumerate(NOISE_LEVELS):
        vals = [results[(p, noise)].success_rate * 100 for p in TRAIN_TYPES]
        bars = ax.bar(
            x + offsets[j],
            vals,
            width,
            label=NOISE_LABELS[noise],
            color=NOISE_COLORS[noise],
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Training type")
    ax.set_title(
        f"Success rate by control noise ({NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_lines(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    x = np.arange(len(NOISE_LEVELS))
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in TRAIN_TYPES:
        vals = [results[(policy, n)].success_rate * 100 for n in NOISE_LEVELS]
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2,
            markersize=8,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )

    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Control noise")
    ax.set_title(
        f"Success vs control noise ({NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_delta(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    """Change in success rate from low → high noise (percentage points)."""
    deltas = [
        (results[(p, "high")].success_rate - results[(p, "low")].success_rate) * 100
        for p in TRAIN_TYPES
    ]
    colors = ["#27ae60" if d >= 0 else "#c0392b" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([POLICY_LABELS[p] for p in TRAIN_TYPES], deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ success rate (high − low, pp)")
    ax.set_title(
        f"Impact of increasing noise ({NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS})"
    )
    ax.grid(axis="y", alpha=0.3)

    for bar, d in zip(bars, deltas):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.3 if y >= 0 else -0.3
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + offset,
            f"{d:+.1f}",
            ha="center",
            va=va,
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_pies(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    fig, axes = plt.subplots(len(TRAIN_TYPES), len(NOISE_LEVELS), figsize=(11, 10))

    for i, policy in enumerate(TRAIN_TYPES):
        for j, noise in enumerate(NOISE_LEVELS):
            ax = axes[i, j]
            r = results[(policy, noise)]
            success = r.episodes - r.failures_total
            fail_sizes = [r.failure_counts[k] for k in FAILURE_KEYS]
            sizes = [success] + fail_sizes
            labels = ["Success", "Obstacle", "Wall", "Timeout"]
            colors = [
                "#2ecc71",
                FAILURE_COLORS["obstacle"],
                FAILURE_COLORS["wall"],
                FAILURE_COLORS["timeout"],
            ]

            wedges, _ = ax.pie(
                sizes,
                labels=None,
                colors=colors,
                startangle=90,
                wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
            )
            if i == 0:
                ax.set_title(NOISE_LABELS[noise], fontsize=10)
            if j == 0:
                ax.set_ylabel(POLICY_LABELS[policy], fontsize=9)

            if i == 0 and j == len(NOISE_LEVELS) - 1:
                ax.legend(
                    wedges,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1.0),
                    fontsize=8,
                )

    fig.suptitle(
        f"Episode outcomes (1000 each; {NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS})",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_stacked_bars(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    labels: List[str] = []
    obstacle_pct: List[float] = []
    wall_pct: List[float] = []
    timeout_pct: List[float] = []

    for policy in TRAIN_TYPES:
        for noise in NOISE_LEVELS:
            r = results[(policy, noise)]
            labels.append(f"{POLICY_LABELS[policy]}\n{NOISE_LABELS[noise]}")
            rates = r.failure_rates
            obstacle_pct.append(rates["obstacle"] * 100)
            wall_pct.append(rates["wall"] * 100)
            timeout_pct.append(rates["timeout"] * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x, obstacle_pct, label="Obstacle", color=FAILURE_COLORS["obstacle"])
    ax.bar(x, wall_pct, bottom=obstacle_pct, label="Wall", color=FAILURE_COLORS["wall"])
    bottom = np.array(obstacle_pct) + np.array(wall_pct)
    ax.bar(x, timeout_pct, bottom=bottom, label="Timeout", color=FAILURE_COLORS["timeout"])

    ax.set_ylabel("Failure rate (% of episodes)")
    ax.set_title(
        f"Failure mode breakdown ({NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_composition(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    labels: List[str] = []
    stacks: Dict[str, List[float]] = {k: [] for k in FAILURE_KEYS}

    for policy in TRAIN_TYPES:
        for noise in NOISE_LEVELS:
            r = results[(policy, noise)]
            labels.append(f"{POLICY_LABELS[policy]}\n{NOISE_LABELS[noise]}")
            total = max(r.failures_total, 1)
            for k in FAILURE_KEYS:
                stacks[k].append(r.failure_counts[k] / total * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(labels))
    for k in FAILURE_KEYS:
        ax.bar(x, stacks[k], bottom=bottom, label=k.capitalize(), color=FAILURE_COLORS[k])
        bottom += np.array(stacks[k])

    ax.set_ylabel("Share of failures (%)")
    ax.set_title(
        f"Failure composition conditional on failure ({NUM_AGENTS}A/{NUM_OBSTACLES}O)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_heatmap(
    results: Dict[Tuple[str, str], EvalResult],
    out_path: Path,
    failure_key: str = "timeout",
) -> None:
    mat = np.zeros((len(NOISE_LEVELS), len(TRAIN_TYPES)))
    for i, noise in enumerate(NOISE_LEVELS):
        for j, policy in enumerate(TRAIN_TYPES):
            mat[i, j] = results[(policy, noise)].failure_rates[failure_key] * 100.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(15, mat.max() * 1.05), aspect="auto")

    ax.set_xticks(range(len(TRAIN_TYPES)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_yticks(range(len(NOISE_LEVELS)))
    ax.set_yticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
    ax.set_xlabel("Training type")
    ax.set_ylabel("Control noise")
    ax.set_title(
        f"{failure_key.capitalize()} failure rate (%) — "
        f"{NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS}"
    )

    for i in range(len(NOISE_LEVELS)):
        for j in range(len(TRAIN_TYPES)):
            val = mat[i, j]
            color = "white" if val > 12 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{failure_key.capitalize()} failure rate (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_reward_bars(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    x = np.arange(len(TRAIN_TYPES))
    width = 0.25
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, noise in enumerate(NOISE_LEVELS):
        vals = [results[(p, noise)].mean_reward for p in TRAIN_TYPES]
        ax.bar(
            x + offsets[j],
            vals,
            width,
            label=NOISE_LABELS[noise],
            color=NOISE_COLORS[noise],
        )

    ax.set_ylabel("Mean episode reward")
    ax.set_xlabel("Training type")
    ax.set_title(
        f"Mean reward by control noise ({NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_reward_lines(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    x = np.arange(len(NOISE_LEVELS))
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in TRAIN_TYPES:
        vals = [results[(policy, n)].mean_reward for n in NOISE_LEVELS]
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2,
            markersize=8,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )

    ax.set_ylabel("Mean episode reward")
    ax.set_xlabel("Control noise")
    ax.set_title(
        f"Mean reward vs control noise ({NUM_AGENTS}A/{NUM_OBSTACLES}O, {DYNAMICS})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[n] for n in NOISE_LEVELS])
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    header = (
        "policy,dynamics,agents,obstacles,control_noise,episodes,"
        "success_rate_pct,mean_reward,failures_total,"
        "fail_obstacle,fail_wall,fail_timeout,training_run,source_file\n"
    )
    rows = []
    for policy in TRAIN_TYPES:
        for noise in NOISE_LEVELS:
            r = results[(policy, noise)]
            rows.append(
                f"{policy},{r.dynamics},{r.num_agents},{r.num_obstacles},{r.control_noise},"
                f"{r.episodes},{r.success_rate * 100:.2f},{r.mean_reward:.2f},"
                f"{r.failures_total},{r.failure_counts['obstacle']},"
                f"{r.failure_counts['wall']},{r.failure_counts['timeout']},"
                f"{r.training_run},{r.source_file}\n"
            )
    out_path.write_text(header + "".join(rows))


def print_summary_table(results: Dict[Tuple[str, str], EvalResult]) -> None:
    print(
        f"\nNoise comparison ({NUM_AGENTS} agent, {NUM_OBSTACLES} obstacles, "
        f"{DYNAMICS}):\n"
    )
    print(
        f"{'Policy':<14} {'Noise':<10} {'Success':>8} "
        f"{'Obstacle':>10} {'Wall':>8} {'Timeout':>9}"
    )
    print("-" * 62)
    for policy in TRAIN_TYPES:
        for noise in NOISE_LEVELS:
            r = results[(policy, noise)]
            fr = r.failure_rates
            print(
                f"{POLICY_LABELS[policy]:<14} {NOISE_LABELS[noise]:<10} "
                f"{r.success_rate * 100:>7.1f}% "
                f"{fr['obstacle'] * 100:>9.2f}% {fr['wall'] * 100:>7.2f}% "
                f"{fr['timeout'] * 100:>8.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot control noise comparison across policies."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing results_<policy>/ folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots/noise"),
        help="Directory for saved plots",
    )
    parser.add_argument("--dynamics", type=str, default=DYNAMICS)
    parser.add_argument("--num-agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--num-obstacles", type=int, default=NUM_OBSTACLES)
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = discover_results(
        results_root,
        dynamics=args.dynamics,
        num_agents=args.num_agents,
        num_obstacles=args.num_obstacles,
    )
    require_complete(results)
    print_summary_table(results)

    plots = {
        "success_rate_heatmap.png": plot_success_heatmap,
        "success_rate_bars.png": plot_success_bars,
        "success_rate_lines.png": plot_success_lines,
        "success_delta_high_vs_low.png": plot_success_delta,
        "failure_pies.png": plot_failure_pies,
        "failure_stacked_bars.png": plot_failure_stacked_bars,
        "failure_composition.png": plot_failure_composition,
        "timeout_failure_heatmap.png": lambda r, p: plot_failure_heatmap(r, p, "timeout"),
        "obstacle_failure_heatmap.png": lambda r, p: plot_failure_heatmap(
            r, p, "obstacle"
        ),
        "mean_reward_bars.png": plot_mean_reward_bars,
        "mean_reward_lines.png": plot_mean_reward_lines,
    }

    for filename, fn in plots.items():
        path = out_dir / filename
        fn(results, path)
        print(f"Saved {path}")

    csv_path = out_dir / "noise_summary.csv"
    write_summary_csv(results, csv_path)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
