#!/usr/bin/env python3
"""
Compare agent/obstacle grid (Suite 2) across the four policy types.

Parses evaluation summary .txt files under results/ and generates plots under
results/plots/agents/ by default.

Suite 2 filter: dynamic dynamics, low control noise, agents+obstacles in
{(5,0), (4,1), (3,2), (2,3), (1,4)} (sum = 5). Excludes the Suite 1 (1, 3) case.
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

# (num_agents, num_obstacles) — display order: more agents → fewer
AGENT_OBSTACLE_CONFIGS: Tuple[Tuple[int, int], ...] = (
    (5, 0),
    (4, 1),
    (3, 2),
    (2, 3),
    (1, 4),
)

POLICY_COLORS = {
    "naive": "#95a5a6",
    "cbf": "#2ecc71",
    "reward_only": "#9b59b6",
    "filter_only": "#e67e22",
}


def config_label(agents: int, obstacles: int) -> str:
    return f"{agents}A / {obstacles}O"


def discover_results(
    results_root: Path,
    dynamics: str = "dynamic",
    control_noise: str = "low",
) -> Dict[Tuple[str, int, int], EvalResult]:
    """One result per (policy, agents, obstacles), preferring the newest file."""
    allowed = set(AGENT_OBSTACLE_CONFIGS)
    candidates: Dict[Tuple[str, int, int], List[EvalResult]] = {}

    for path in sorted(results_root.glob("results_*/*/*.txt")):
        parsed = parse_result_file(path)
        if parsed is None:
            continue
        if parsed.dynamics != dynamics:
            continue
        if parsed.control_noise != control_noise:
            continue
        if parsed.policy not in TRAIN_TYPES:
            continue
        key_ao = (parsed.num_agents, parsed.num_obstacles)
        if key_ao not in allowed:
            continue

        key = (parsed.policy, parsed.num_agents, parsed.num_obstacles)
        candidates.setdefault(key, []).append(parsed)

    selected: Dict[Tuple[str, int, int], EvalResult] = {}
    for key, items in candidates.items():
        items.sort(key=lambda r: r.source_file)
        if len(items) > 1:
            print(
                f"Warning: multiple matches for {key[0]} {key[1]}A/{key[2]}O; "
                f"using {items[-1].source_file}"
            )
        selected[key] = items[-1]
    return selected


def require_complete(results: Dict[Tuple[str, int, int], EvalResult]) -> None:
    missing = [
        (policy, a, o)
        for policy in TRAIN_TYPES
        for a, o in AGENT_OBSTACLE_CONFIGS
        if (policy, a, o) not in results
    ]
    if missing:
        lines = [
            f"  - {POLICY_LABELS[p]} / {config_label(a, o)}"
            for p, a, o in missing
        ]
        raise SystemExit(
            "Missing Suite 2 results (dynamic, low noise, agents+obstacles=5):\n"
            + "\n".join(lines)
        )


def success_matrix(results: Dict[Tuple[str, int, int], EvalResult]) -> np.ndarray:
    """Rows = configs, cols = policies."""
    mat = np.zeros((len(AGENT_OBSTACLE_CONFIGS), len(TRAIN_TYPES)))
    for i, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
        for j, policy in enumerate(TRAIN_TYPES):
            mat[i, j] = results[(policy, a, o)].success_rate * 100.0
    return mat


def plot_success_heatmap(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    mat = success_matrix(results)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(TRAIN_TYPES)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_yticks(range(len(AGENT_OBSTACLE_CONFIGS)))
    ax.set_yticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
    ax.set_xlabel("Training type")
    ax.set_ylabel("Agents / obstacles")
    ax.set_title("Success rate (%) — dynamic, low noise")

    for i in range(len(AGENT_OBSTACLE_CONFIGS)):
        for j in range(len(TRAIN_TYPES)):
            val = mat[i, j]
            color = "white" if val < 45 or val > 85 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_bars(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    n_configs = len(AGENT_OBSTACLE_CONFIGS)
    n_policies = len(TRAIN_TYPES)
    x = np.arange(n_configs)
    width = 0.18
    offsets = np.linspace(-(n_policies - 1) / 2, (n_policies - 1) / 2, n_policies) * width

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for j, policy in enumerate(TRAIN_TYPES):
        vals = [
            results[(policy, a, o)].success_rate * 100
            for a, o in AGENT_OBSTACLE_CONFIGS
        ]
        bars = ax.bar(
            x + offsets[j],
            vals,
            width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
        for bar in bars:
            h = bar.get_height()
            if h > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.8,
                    f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Agents / obstacles (sum = 5)")
    ax.set_title("Success rate by environment density (dynamic, low noise)")
    ax.set_xticks(x)
    ax.set_xticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_lines(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    x = np.arange(len(AGENT_OBSTACLE_CONFIGS))
    fig, ax = plt.subplots(figsize=(9, 5))

    for policy in TRAIN_TYPES:
        vals = [
            results[(policy, a, o)].success_rate * 100
            for a, o in AGENT_OBSTACLE_CONFIGS
        ]
        ax.plot(
            x,
            vals,
            marker="o",
            linewidth=2,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )

    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Agents / obstacles (more agents → fewer)")
    ax.set_title("Success vs agent/obstacle mix (dynamic, low noise)")
    ax.set_xticks(x)
    ax.set_xticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_pies(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    fig, axes = plt.subplots(len(TRAIN_TYPES), len(AGENT_OBSTACLE_CONFIGS), figsize=(16, 10))

    for i, policy in enumerate(TRAIN_TYPES):
        for j, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
            ax = axes[i, j]
            r = results[(policy, a, o)]
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
                ax.set_title(config_label(a, o), fontsize=9)
            if j == 0:
                ax.set_ylabel(POLICY_LABELS[policy], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0 and j == len(AGENT_OBSTACLE_CONFIGS) - 1:
                ax.legend(
                    wedges,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1.0),
                    fontsize=8,
                )

    fig.suptitle(
        "Episode outcomes (1000 episodes each; dynamic, low noise)",
        y=1.01,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_stacked_bars(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    labels: List[str] = []
    obstacle_pct: List[float] = []
    wall_pct: List[float] = []
    timeout_pct: List[float] = []

    for policy in TRAIN_TYPES:
        for a, o in AGENT_OBSTACLE_CONFIGS:
            r = results[(policy, a, o)]
            labels.append(f"{POLICY_LABELS[policy]}\n{config_label(a, o)}")
            rates = r.failure_rates
            obstacle_pct.append(rates["obstacle"] * 100)
            wall_pct.append(rates["wall"] * 100)
            timeout_pct.append(rates["timeout"] * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(x, obstacle_pct, label="Obstacle", color=FAILURE_COLORS["obstacle"])
    ax.bar(x, wall_pct, bottom=obstacle_pct, label="Wall", color=FAILURE_COLORS["wall"])
    bottom = np.array(obstacle_pct) + np.array(wall_pct)
    ax.bar(x, timeout_pct, bottom=bottom, label="Timeout", color=FAILURE_COLORS["timeout"])

    ax.set_ylabel("Failure rate (% of episodes)")
    ax.set_title("Failure mode breakdown (dynamic, low noise)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_composition(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    labels: List[str] = []
    stacks: Dict[str, List[float]] = {k: [] for k in FAILURE_KEYS}

    for policy in TRAIN_TYPES:
        for a, o in AGENT_OBSTACLE_CONFIGS:
            r = results[(policy, a, o)]
            labels.append(f"{POLICY_LABELS[policy]}\n{config_label(a, o)}")
            total = max(r.failures_total, 1)
            for k in FAILURE_KEYS:
                stacks[k].append(r.failure_counts[k] / total * 100)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(16, 5))
    bottom = np.zeros(len(labels))
    for k in FAILURE_KEYS:
        ax.bar(x, stacks[k], bottom=bottom, label=k.capitalize(), color=FAILURE_COLORS[k])
        bottom += np.array(stacks[k])

    ax.set_ylabel("Share of failures (%)")
    ax.set_title("Failure composition conditional on failure (dynamic, low noise)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_reward_bars(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    n_configs = len(AGENT_OBSTACLE_CONFIGS)
    n_policies = len(TRAIN_TYPES)
    x = np.arange(n_configs)
    width = 0.18
    offsets = np.linspace(-(n_policies - 1) / 2, (n_policies - 1) / 2, n_policies) * width

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for j, policy in enumerate(TRAIN_TYPES):
        vals = [results[(policy, a, o)].mean_reward for a, o in AGENT_OBSTACLE_CONFIGS]
        ax.bar(
            x + offsets[j],
            vals,
            width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_ylabel("Mean episode reward")
    ax.set_xlabel("Agents / obstacles (sum = 5)")
    ax.set_title(
        "Mean reward by environment density (dynamic, low noise)\n"
        "(negative values: CBF penalty-dominated episodes)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_heatmap(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    """Heatmap of obstacle-collision failure rate (rows=configs, cols=policies)."""
    mat = np.zeros((len(AGENT_OBSTACLE_CONFIGS), len(TRAIN_TYPES)))
    for i, (a, o) in enumerate(AGENT_OBSTACLE_CONFIGS):
        for j, policy in enumerate(TRAIN_TYPES):
            mat[i, j] = results[(policy, a, o)].failure_rates["obstacle"] * 100.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(40, mat.max() * 1.05), aspect="auto")

    ax.set_xticks(range(len(TRAIN_TYPES)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_yticks(range(len(AGENT_OBSTACLE_CONFIGS)))
    ax.set_yticklabels([config_label(a, o) for a, o in AGENT_OBSTACLE_CONFIGS])
    ax.set_xlabel("Training type")
    ax.set_ylabel("Agents / obstacles")
    ax.set_title("Obstacle collision rate (%) — dynamic, low noise")

    for i in range(len(AGENT_OBSTACLE_CONFIGS)):
        for j in range(len(TRAIN_TYPES)):
            val = mat[i, j]
            color = "white" if val > 20 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Obstacle failure rate (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    results: Dict[Tuple[str, int, int], EvalResult], out_path: Path
) -> None:
    header = (
        "policy,dynamics,agents,obstacles,control_noise,episodes,"
        "success_rate_pct,mean_reward,failures_total,"
        "fail_obstacle,fail_wall,fail_timeout,training_run,source_file\n"
    )
    rows = []
    for policy in TRAIN_TYPES:
        for a, o in AGENT_OBSTACLE_CONFIGS:
            r = results[(policy, a, o)]
            rows.append(
                f"{policy},{r.dynamics},{r.num_agents},{r.num_obstacles},{r.control_noise},"
                f"{r.episodes},{r.success_rate * 100:.2f},{r.mean_reward:.2f},"
                f"{r.failures_total},{r.failure_counts['obstacle']},"
                f"{r.failure_counts['wall']},{r.failure_counts['timeout']},"
                f"{r.training_run},{r.source_file}\n"
            )
    out_path.write_text(header + "".join(rows))


def print_summary_table(results: Dict[Tuple[str, int, int], EvalResult]) -> None:
    print("\nSuite 2 results (dynamic, low noise, agents+obstacles=5):\n")
    print(
        f"{'Policy':<14} {'Config':<10} {'Success':>8} "
        f"{'Obstacle':>10} {'Wall':>8} {'Timeout':>9}"
    )
    print("-" * 62)
    for policy in TRAIN_TYPES:
        for a, o in AGENT_OBSTACLE_CONFIGS:
            r = results[(policy, a, o)]
            fr = r.failure_rates
            print(
                f"{POLICY_LABELS[policy]:<14} {config_label(a, o):<10} "
                f"{r.success_rate * 100:>7.1f}% "
                f"{fr['obstacle'] * 100:>9.2f}% {fr['wall'] * 100:>7.2f}% "
                f"{fr['timeout'] * 100:>8.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Suite 2 agent/obstacle grid comparison."
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
        default=Path("results/plots/agents"),
        help="Directory for saved plots",
    )
    parser.add_argument("--dynamics", type=str, default="dynamic")
    parser.add_argument("--control-noise", type=str, default="low")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = discover_results(
        results_root,
        dynamics=args.dynamics,
        control_noise=args.control_noise,
    )
    require_complete(results)
    print_summary_table(results)

    plots = {
        "success_rate_heatmap.png": plot_success_heatmap,
        "success_rate_bars.png": plot_success_bars,
        "success_rate_lines.png": plot_success_lines,
        "failure_pies.png": plot_failure_pies,
        "failure_stacked_bars.png": plot_failure_stacked_bars,
        "failure_composition.png": plot_failure_composition,
        "obstacle_failure_heatmap.png": plot_failure_heatmap,
        "mean_reward_bars.png": plot_mean_reward_bars,
    }

    for filename, fn in plots.items():
        path = out_dir / filename
        fn(results, path)
        print(f"Saved {path}")

    csv_path = out_dir / "suite2_summary.csv"
    write_summary_csv(results, csv_path)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
