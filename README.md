# CBF Learning Demo

This repository contains a demonstration of combining Reinforcement Learning (RL) with Control Barrier Functions (CBF) for robot navigation, using `rsl_rl` and IsaacGym-like vectorized environments. 

## Installation

The project uses Conda to manage its dependencies. First, ensure you have Conda installed on your system.
Create and activate the conda environment by running:
```bash
conda env create -f environment.yml
conda activate cbf_learning
```

## Environments Setup

The project provides several environment configurations through the `UnifiedNavigationEnv` by passing different arguments:
- **Naive**: Standard RL without CBF.
- **CBF (Hybrid)**: RL with CBF action filtering and CBF reward penalties.
- **Filter Only**: RL with CBF action filtering but no reward penalties.
- **Reward Only (Soft CBF)**: RL with CBF reward penalties but no action filtering.

## Training

You can train the agent using the provided bash scripts. For example, to run headless training:
```bash
# Train Naive 
./train_naive.sh

# Train CBF
./train_cbf.sh

# Train Filter Only
./train_filter_only.sh

# Train Reward Only
./train_reward_only.sh
```

Alternatively, you can run `train.py` directly with custom command line flags. For example:
```bash
python train.py --env cbf --use_cbf_action_filtering --use_cbf_reward_penalty --headless
```

Training logs and checkpoints are automatically saved to `logs/navigation_<env_type>/`.

## Testing / Evaluation

After training, evaluate the learned policy using the test scripts:
```bash
./test_naive.sh
./test_cbf.sh
./test_filter_only.sh
./test_reward_only.sh
```

Or run `test.py` directly. For example:
```bash
python test.py --env cbf --use_cbf_action_filtering --use_cbf_reward_penalty --dynamics_model quasi_static
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

Here are the specific files that were run on thorax GPU:

``` bash
python plot_tb_reward_log_steps.py     --naive logs/navigation_naive/navigation_naive_20260514_074524/events.out.tfevents.1778769928.thorax.238343.0     --cbf logs/navigation_cbf/navigation_cbf_20260514_084143/events.out.tfevents.1778773306.thorax.256096.0     --only-cbf logs/navigation_filter_only/navigation_filter_only_20260514_100151/events.out.tfevents.1778778114.thorax.275936.0     --soft-cbf logs/navigation_reward_only/navigation_reward_only_20260514_104520/events.out.tfevents.1778780723.thorax.291840.0     --out-main logs/plots/comparison_results.pdf
```

Here are the specific files that were run on thorax CPU:
``` bash
python plot_tb_reward_log_steps.py --naive logs/navigation_naive/navigation_naive_20260524_100904/events.out.tfevents.1779642548.thorax.661641.0 --cbf logs/navigation_cbf/navigation_cbf_20260524_121158/events.out.tfevents.1779649921.thorax.701411.0 --only-cbf logs/navigation_filter_only/navigation_filter_only_20260524_121421/events.out.tfevents.1779650065.thorax.703063.0 --soft-cbf logs/navigation_reward_only/navigation_reward_only_20260524_121327/events.out.tfevents.1779650011.thorax.702285.0 --out-main logs/plots/comparison_results_cpu.pdf
```


This generates PDF plots of the training rewards and obstacle collisions over time. The main plot is saved to `logs/plots/mean_episode_reward.pdf` and a summary bar chart to `logs/plots/mean_episode_reward_summary.pdf`.

How to see the logs of the training process:
```tensorboard --logdir logs/ --port 6006```