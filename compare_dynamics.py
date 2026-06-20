#!/usr/bin/env python3
"""
Compare quasi-static vs dynamic training across the four policy types (Suite 1).

Parses evaluation summary .txt files under results/ and generates plots under
results/plots/dynamics/ by default.

Suite 1 filter: 1 agent, 3 obstacles, low control noise (standard training grid).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

TRAIN_TYPES = ("naive", "cbf", "reward_only", "filter_only")
DYNAMICS = ("quasi_static", "dynamic")
FAILURE_KEYS = ("obstacle", "wall", "timeout")

POLICY_LABELS = {
    "naive": "Naive",
    "cbf": "CBF",
    "reward_only": "Reward Only",
    "filter_only": "Filter Only",
}

DYNAMICS_LABELS = {
    "quasi_static": "Quasi-static",
    "dynamic": "Dynamic",
}

FAILURE_COLORS = {
    "obstacle": "#e74c3c",
    "wall": "#3498db",
    "timeout": "#f39c12",
}

SUMMARY_RE = re.compile(r"=== Evaluation Summary: (\w+) ===")
FIELD_RES = {
    "training_run": re.compile(r"^Training run:\s+(\S+)"),
    "dynamics": re.compile(r"^Dynamics:\s+(\S+)"),
    "control_noise": re.compile(r"^Control noise:\s+(\S+)"),
    "agents": re.compile(r"^Agents:\s+(\d+)"),
    "obstacles": re.compile(r"^Obstacles:\s+(\d+)"),
    "episodes": re.compile(r"^Episodes:\s+(\d+)"),
    "mean_reward": re.compile(r"^Mean Reward:\s+(-?[\d.]+)"),
    "success_rate": re.compile(r"^Success Rate:\s+([\d.]+)%"),
    "failures_total": re.compile(r"^Failures:\s+(\d+)"),
    "failure_obstacle": re.compile(r"^ - Obstacle:\s+(\d+)"),
    "failure_wall": re.compile(r"^ - Wall:\s+(\d+)"),
    "failure_timeout": re.compile(r"^ - Timeout:\s+(\d+)"),
}


@dataclass
class EvalResult:
    policy: str
    dynamics: str
    num_agents: int
    num_obstacles: int
    control_noise: str
    episodes: int
    success_rate: float
    mean_reward: float
    failures_total: int
    failure_counts: Dict[str, int] = field(default_factory=dict)
    training_run: str = ""
    source_file: str = ""

    @property
    def failure_rates(self) -> Dict[str, float]:
        if self.episodes <= 0:
            return {k: 0.0 for k in FAILURE_KEYS}
        return {k: self.failure_counts.get(k, 0) / self.episodes for k in FAILURE_KEYS}


def _policy_from_summary(summary: str) -> Optional[str]:
    mapping = {
        "NAIVE": "naive",
        "CBF": "cbf",
        "REWARD_ONLY": "reward_only",
        "FILTER_ONLY": "filter_only",
    }
    return mapping.get(summary.upper())


def _policy_from_path(path: Path) -> Optional[str]:
    name = path.as_posix()
    for policy in TRAIN_TYPES:
        if f"results_{policy}" in name or f"navigation_{policy}" in name:
            return policy
    return None


def parse_result_file(path: Path) -> Optional[EvalResult]:
    text = path.read_text()
    lines = text.splitlines()

    policy: Optional[str] = None
    fields: Dict[str, str] = {}

    for line in lines:
        m = SUMMARY_RE.match(line)
        if m:
            policy = _policy_from_summary(m.group(1))
            continue
        for key, pattern in FIELD_RES.items():
            m = pattern.match(line)
            if m:
                fields[key] = m.group(1)

    if policy is None:
        policy = _policy_from_path(path)
    if policy is None or "dynamics" not in fields:
        return None

    try:
        return EvalResult(
            policy=policy,
            dynamics=fields["dynamics"],
            num_agents=int(fields.get("agents", 0)),
            num_obstacles=int(fields.get("obstacles", 0)),
            control_noise=fields.get("control_noise", "low"),
            episodes=int(fields.get("episodes", 0)),
            success_rate=float(fields.get("success_rate", 0.0)) / 100.0,
            mean_reward=float(fields.get("mean_reward", 0.0)),
            failures_total=int(fields.get("failures_total", 0)),
            failure_counts={
                "obstacle": int(fields.get("failure_obstacle", 0)),
                "wall": int(fields.get("failure_wall", 0)),
                "timeout": int(fields.get("failure_timeout", 0)),
            },
            training_run=fields.get("training_run", ""),
            source_file=str(path),
        )
    except (KeyError, ValueError):
        return None


def discover_results(
    results_root: Path,
    num_agents: int = 1,
    num_obstacles: int = 3,
    control_noise: str = "low",
) -> Dict[Tuple[str, str], EvalResult]:
    """Return one result per (policy, dynamics), preferring the newest file."""
    candidates: Dict[Tuple[str, str], List[EvalResult]] = {}

    for path in sorted(results_root.glob("results_*/*/*.txt")):
        parsed = parse_result_file(path)
        if parsed is None:
            continue
        if parsed.num_agents != num_agents:
            continue
        if parsed.num_obstacles != num_obstacles:
            continue
        if parsed.control_noise != control_noise:
            continue
        if parsed.dynamics not in DYNAMICS:
            continue
        if parsed.policy not in TRAIN_TYPES:
            continue

        key = (parsed.policy, parsed.dynamics)
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
        (policy, dyn)
        for policy in TRAIN_TYPES
        for dyn in DYNAMICS
        if (policy, dyn) not in results
    ]
    if missing:
        lines = [f"  - {POLICY_LABELS[p]} / {DYNAMICS_LABELS[d]}" for p, d in missing]
        raise SystemExit(
            "Missing Suite 1 results (1 agent, 3 obstacles, low noise):\n"
            + "\n".join(lines)
        )


def success_matrix(results: Dict[Tuple[str, str], EvalResult]) -> np.ndarray:
    mat = np.zeros((len(DYNAMICS), len(TRAIN_TYPES)))
    for j, policy in enumerate(TRAIN_TYPES):
        for i, dyn in enumerate(DYNAMICS):
            mat[i, j] = results[(policy, dyn)].success_rate * 100.0
    return mat


def plot_success_heatmap(results: Dict[Tuple[str, str], EvalResult], out_path: Path) -> None:
    mat = success_matrix(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(TRAIN_TYPES)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_yticks(range(len(DYNAMICS)))
    ax.set_yticklabels([DYNAMICS_LABELS[d] for d in DYNAMICS])
    ax.set_xlabel("Training type")
    ax.set_ylabel("Dynamics model")
    ax.set_title("Success rate (%) — quasi-static vs dynamic")

    for i in range(len(DYNAMICS)):
        for j in range(len(TRAIN_TYPES)):
            val = mat[i, j]
            color = "white" if val < 45 or val > 85 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_bars(results: Dict[Tuple[str, str], EvalResult], out_path: Path) -> None:
    x = np.arange(len(TRAIN_TYPES))
    width = 0.36
    qs_vals = [results[(p, "quasi_static")].success_rate * 100 for p in TRAIN_TYPES]
    dyn_vals = [results[(p, "dynamic")].success_rate * 100 for p in TRAIN_TYPES]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_qs = ax.bar(x - width / 2, qs_vals, width, label=DYNAMICS_LABELS["quasi_static"], color="#5dade2")
    bars_dyn = ax.bar(x + width / 2, dyn_vals, width, label=DYNAMICS_LABELS["dynamic"], color="#1a5276")

    ax.set_ylabel("Success rate (%)")
    ax.set_xlabel("Training type")
    ax.set_title("Success rate by training type and dynamics model")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars_qs, bars_dyn):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_pies(results: Dict[Tuple[str, str], EvalResult], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    configs = [(p, d) for d in DYNAMICS for p in TRAIN_TYPES]

    for ax, (policy, dyn) in zip(axes.flat, configs):
        r = results[(policy, dyn)]
        success = r.episodes - r.failures_total
        fail_sizes = [r.failure_counts[k] for k in FAILURE_KEYS]
        sizes = [success] + fail_sizes
        labels = ["Success", "Obstacle", "Wall", "Timeout"]
        colors = ["#2ecc71", FAILURE_COLORS["obstacle"], FAILURE_COLORS["wall"], FAILURE_COLORS["timeout"]]

        wedges, _ = ax.pie(
            sizes,
            labels=None,
            colors=colors,
            startangle=90,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        ax.set_title(
            f"{POLICY_LABELS[policy]}\n{DYNAMICS_LABELS[dyn]}",
            fontsize=10,
        )
        if policy == TRAIN_TYPES[0] and dyn == DYNAMICS[0]:
            ax.legend(wedges, labels, loc="upper left", bbox_to_anchor=(-0.35, 1.05), fontsize=8)

    fig.suptitle("Episode outcomes (1000 episodes each)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_stacked_bars(results: Dict[Tuple[str, str], EvalResult], out_path: Path) -> None:
    labels: List[str] = []
    obstacle_pct: List[float] = []
    wall_pct: List[float] = []
    timeout_pct: List[float] = []

    for policy in TRAIN_TYPES:
        for dyn in DYNAMICS:
            r = results[(policy, dyn)]
            labels.append(f"{POLICY_LABELS[policy]}\n{DYNAMICS_LABELS[dyn]}")
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
    ax.set_title("Failure mode breakdown (among all episodes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_composition_among_failures(
    results: Dict[Tuple[str, str], EvalResult], out_path: Path
) -> None:
    """Stacked bars showing how failures split among modes (100% = all failures)."""
    labels: List[str] = []
    stacks: Dict[str, List[float]] = {k: [] for k in FAILURE_KEYS}

    for policy in TRAIN_TYPES:
        for dyn in DYNAMICS:
            r = results[(policy, dyn)]
            labels.append(f"{POLICY_LABELS[policy]}\n{DYNAMICS_LABELS[dyn]}")
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
    ax.set_title("Failure composition (conditional on failure)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_delta(results: Dict[Tuple[str, str], EvalResult], out_path: Path) -> None:
    deltas = [
        (results[(p, "dynamic")].success_rate - results[(p, "quasi_static")].success_rate) * 100
        for p in TRAIN_TYPES
    ]
    colors = ["#27ae60" if d >= 0 else "#c0392b" for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([POLICY_LABELS[p] for p in TRAIN_TYPES], deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Δ success rate (dynamic − quasi-static, pp)")
    ax.set_title("Effect of dynamic vs quasi-static training")
    ax.grid(axis="y", alpha=0.3)

    for bar, d in zip(bars, deltas):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.3 if y >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset, f"{d:+.1f}", ha="center", va=va, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mean_reward_bars(results: Dict[Tuple[str, str], EvalResult], out_path: Path) -> None:
    x = np.arange(len(TRAIN_TYPES))
    width = 0.36
    qs_vals = [results[(p, "quasi_static")].mean_reward for p in TRAIN_TYPES]
    dyn_vals = [results[(p, "dynamic")].mean_reward for p in TRAIN_TYPES]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, qs_vals, width, label=DYNAMICS_LABELS["quasi_static"], color="#5dade2")
    ax.bar(x + width / 2, dyn_vals, width, label=DYNAMICS_LABELS["dynamic"], color="#1a5276")

    ax.set_ylabel("Mean episode reward")
    ax.set_xlabel("Training type")
    ax.set_title("Mean reward by training type and dynamics model")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in TRAIN_TYPES])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(results: Dict[Tuple[str, str], EvalResult], out_path: Path) -> None:
    header = (
        "policy,dynamics,agents,obstacles,control_noise,episodes,"
        "success_rate_pct,mean_reward,failures_total,"
        "fail_obstacle,fail_wall,fail_timeout,training_run,source_file\n"
    )
    rows = []
    for policy in TRAIN_TYPES:
        for dyn in DYNAMICS:
            r = results[(policy, dyn)]
            rows.append(
                f"{policy},{dyn},{r.num_agents},{r.num_obstacles},{r.control_noise},"
                f"{r.episodes},{r.success_rate * 100:.2f},{r.mean_reward:.2f},"
                f"{r.failures_total},{r.failure_counts['obstacle']},"
                f"{r.failure_counts['wall']},{r.failure_counts['timeout']},"
                f"{r.training_run},{r.source_file}\n"
            )
    out_path.write_text(header + "".join(rows))


def print_summary_table(results: Dict[Tuple[str, str], EvalResult]) -> None:
    print("\nSuite 1 results (1 agent, 3 obstacles, low noise):\n")
    print(f"{'Policy':<14} {'Dynamics':<14} {'Success':>8} {'Obstacle':>10} {'Wall':>8} {'Timeout':>9}")
    print("-" * 65)
    for policy in TRAIN_TYPES:
        for dyn in DYNAMICS:
            r = results[(policy, dyn)]
            fr = r.failure_rates
            print(
                f"{POLICY_LABELS[policy]:<14} {DYNAMICS_LABELS[dyn]:<14} "
                f"{r.success_rate * 100:>7.1f}% "
                f"{fr['obstacle'] * 100:>9.2f}% {fr['wall'] * 100:>7.2f}% {fr['timeout'] * 100:>8.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Suite 1 dynamics comparison.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing results_<policy>/ folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/plots/dynamics"),
        help="Directory for saved plots",
    )
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--num-obstacles", type=int, default=3)
    parser.add_argument("--control-noise", type=str, default="low")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = discover_results(
        results_root,
        num_agents=args.num_agents,
        num_obstacles=args.num_obstacles,
        control_noise=args.control_noise,
    )
    require_complete(results)
    print_summary_table(results)

    plots = {
        "success_rate_heatmap.png": plot_success_heatmap,
        "success_rate_bars.png": plot_success_bars,
        "failure_pies.png": plot_failure_pies,
        "failure_stacked_bars.png": plot_failure_stacked_bars,
        "failure_composition.png": plot_failure_composition_among_failures,
        "success_delta.png": plot_success_delta,
        "mean_reward_bars.png": plot_mean_reward_bars,
    }

    for filename, fn in plots.items():
        path = out_dir / filename
        fn(results, path)
        print(f"Saved {path}")

    csv_path = out_dir / "suite1_summary.csv"
    write_summary_csv(results, csv_path)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
