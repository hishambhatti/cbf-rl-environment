# CBF Learning Demo

This repository contains a demonstration of combining Reinforcement Learning (RL) with Control Barrier Functions (CBF) for robot navigation, using `rsl_rl` and IsaacGym-like vectorized environments.

This repo started as a replication of [lzyang2000/cbf-rl-navigation-demo](https://github.com/lzyang2000/cbf-rl-navigation-demo) and is being extended with multi-agent navigation, double-integrator dynamics (acceleration control), and other experiment flags.

## Installation

The project uses Conda to manage its dependencies. First, ensure you have Conda installed on your system.
Create and activate the conda environment by running:
```bash
conda env create -f environment.yml
conda activate cbf_learning
```

Or use the virtualenv + `requirements.txt` workflow if you prefer.

## Environments Setup

The project provides several environment configurations through the `UnifiedNavigationEnv` by passing different arguments:
- **Naive**: Standard RL without CBF.
- **CBF (Hybrid)**: RL with CBF action filtering and CBF reward penalties.
- **Filter Only**: RL with CBF action filtering but no reward penalties.
- **Reward Only (Soft CBF)**: RL with CBF reward penalties but no action filtering.

## Default behavior (original replication)

**No extra flags = original single-agent, single-integrator replication.**

The default settings match the original [cbf-rl-navigation-demo](https://github.com/lzyang2000/cbf-rl-navigation-demo):
- `obs_layout=legacy` — observation key is `last_velocity`
- `dynamics_model=quasi_static` — action is velocity (single integrator)
- `num_agents=1` — single robot per world

So these commands should work out of the box with old checkpoints:

```bash
CUDA_VISIBLE_DEVICES=3 ./test_naive.sh --headless
CUDA_VISIBLE_DEVICES=3 ./test_cbf.sh --headless
CUDA_VISIBLE_DEVICES=3 ./test_filter_only.sh --headless
CUDA_VISIBLE_DEVICES=3 ./test_reward_only.sh --headless
```

Training with defaults (same as original demo):

```bash
./train_naive.sh --headless
./train_cbf.sh --headless
./train_filter_only.sh --headless
./train_reward_only.sh --headless
```

Each new training run writes an `env_meta.json` next to its checkpoints. `test.py` reads that file automatically so you usually do not need to pass layout/dynamics flags when evaluating a run you just trained.

## Extended experiments (opt-in flags)

Pass these flags to `train.py` or `test.py` (the shell scripts forward `$@`):

| Flag | Values | Default | What it does |
|------|--------|---------|--------------|
| `--dynamics_model` | `quasi_static`, `dynamic` | `quasi_static` | Single-integrator (velocity) vs double-integrator (acceleration) |
| `--obs_layout` | `legacy`, `current` | `legacy` | `last_velocity` vs `robot_vel` obs key (affects flat obs ordering) |
| `--num_agents` | integer ≥ 1 | `1` | Multi-agent: each agent sees other agents as dynamic obstacles |

Examples:

```bash
# Double-integrator (acceleration) training
./train_cbf.sh --headless --dynamics_model dynamic

# Multi-agent training with 2 robots per world
./train_cbf.sh --headless --num_agents 2

# Evaluate a run trained with non-default settings
./test_cbf.sh --headless --dynamics_model dynamic --num_agents 2

# Explicitly match a checkpoint trained before env_meta.json existed
./test_cbf.sh --headless --obs_layout legacy --dynamics_model quasi_static
```

### Why `obs_layout` matters

The flat observation vector passed to the policy is built by **sorting the observation dict keys alphabetically** and concatenating. Renaming `last_velocity` → `robot_vel` changes the feature ordering even when the total dimension stays 9. Old checkpoints expect `legacy` layout:

| Layout | Sorted keys | Use with |
|--------|-------------|----------|
| `legacy` | `goal_pos`, `last_velocity`, `obstacles`, `robot_pos` | Original replication / performance-test checkpoints |
| `current` | `goal_pos`, `obstacles`, `robot_pos`, `robot_vel` | New experiments that opt into the renamed key |

## Training

You can train the agent using the provided bash scripts. For example, to run headless training:
```bash
# Train Naive 
./train_naive.sh --headless

# Train CBF
./train_cbf.sh --headless

# Train Filter Only
./train_filter_only.sh --headless

# Train Reward Only
./train_reward_only.sh --headless
```

Alternatively, you can run `train.py` directly with custom command line flags. For example:
```bash
python train.py --env cbf --use_cbf_action_filtering --use_cbf_reward_penalty --headless
```

Training logs and checkpoints are automatically saved to `logs/navigation_<env_type>/`.

## Testing / Evaluation

After training, evaluate the learned policy using the test scripts:
```bash
./test_naive.sh --headless
./test_cbf.sh --headless
./test_filter_only.sh --headless
./test_reward_only.sh --headless
```

Or run `test.py` directly. For example:
```bash
python test.py --env cbf --use_cbf_action_filtering --use_cbf_reward_penalty --headless
```

The test script automatically finds the latest run directory for the specified environment type and loads the latest checkpoint. It will play out the scenario visually (unless run with `--headless`) and output success rate and failure reasons (e.g., collisions).

## Plotting TensorBoard Logs

The repository includes a script to plot the `Mean episode reward` over training steps from TensorBoard event files. You'll need the paths to your run's event files. 

Example usage:
```bash
python plot_tb_reward_log_steps.py \
    --cbf logs/navigation_cbf/.../events.out.tfevents... \
    --naive logs/navigation_naive/.../events.out.tfevents... \
    --only-cbf logs/navigation_filter_only/.../events.out.tfevents... \
    --soft-cbf logs/navigation_reward_only/.../events.out.tfevents... \
    --out-main logs/plots/mean_episode_reward.pdf
```

This generates PDF plots of the training rewards and obstacle collisions over time. The main plot is saved to `logs/plots/mean_episode_reward.pdf` and a summary bar chart to `logs/plots/mean_episode_reward_summary.pdf`.

How to see the logs of the training process:
```bash
tensorboard --logdir logs/ --port 6006
```

## Branch overview

| Branch | Purpose |
|--------|---------|
| `performance-test` | Known-good baseline matching the original demo |
| `jang-integration` | `performance-test` + multi-agent / dynamics extensions from `jang` |
| `jang` | Partner branch with MARL and acceleration experiments |
| `main` | Integration branch with logs, requirements, and merged fixes |
| `param-study` | Earlier param-study work (obs rename, dynamics flags) |
