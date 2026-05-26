import os
from datetime import datetime
import argparse # Import argparse

# Import configuration
from config import cfg, get_log_dir, FLATTENED_OBS_SIZE

from nav_env.unified_navigation_env import UnifiedNavigationEnv

# Attempt to import rsl_rl components
try:
    from rsl_rl.runners import OnPolicyRunner
    _RSL_RL_AVAILABLE = True
except ImportError:
    print("Warning: rsl-rl components not found. Ensure rsl-rl is installed correctly.")
    print("Training script requires rsl-rl to function.")
    _RSL_RL_AVAILABLE = False

def parse_args():
    """Parses command-line arguments. CLI flags override nav_config.yaml defaults."""
    parser = argparse.ArgumentParser(description="Train navigation agent with rsl-rl")
    parser.add_argument('--env', type=str, default='naive',
                        help="Label for this run (naive, cbf, reward_only, filter_only, …)")
    parser.add_argument("--use_cbf_action_filtering", action="store_true", default=None,
                        help="Override cbf.use_action_filtering from nav_config.yaml")
    parser.add_argument("--use_cbf_reward_penalty", action="store_true", default=None,
                        help="Override cbf.use_reward_penalty from nav_config.yaml")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI).")
    parser.add_argument('--dynamics_model', type=str, default=None,
                        choices=['dynamic', 'quasi_static'],
                        help="Override dynamics.model from nav_config.yaml")
    return parser.parse_args()

def train():
    """Initializes and runs the rsl-rl training process."""
    if not _RSL_RL_AVAILABLE:
        print("Cannot proceed without rsl-rl installed.")
        return

    args = parse_args()

    # CLI flags override YAML defaults; None means "use YAML value"
    use_cbf_action_filtering = (
        args.use_cbf_action_filtering
        if args.use_cbf_action_filtering is not None
        else cfg['cbf']['use_action_filtering']
    )
    use_cbf_reward_penalty = (
        args.use_cbf_reward_penalty
        if args.use_cbf_reward_penalty is not None
        else cfg['cbf']['use_reward_penalty']
    )
    dynamics_model = args.dynamics_model or cfg['env']['dynamics_model']

    # Add environment type and timestamp to run name and experiment name for better tracking
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg['runner']['run_name'] = f"{cfg['runner']['run_name']}_{args.env}_{timestamp}"
    cfg['runner']['experiment_name'] = f"{cfg['runner']['experiment_name']}_{args.env}"

    print("--- Starting Training ---")
    print(f"Config file:  nav_config.yaml")
    print(f"Run Name:     {cfg['runner']['run_name']}")
    print(f"Device:       {cfg['device']}")
    print(f"Agents:       {cfg['env']['num_agents']}, Obstacles: {cfg['env']['num_obstacles']}, World: {cfg['env']['world_size']}m")
    print(f"Dynamics:     {dynamics_model}")
    print(f"CBF filter:   {use_cbf_action_filtering}, CBF penalty: {use_cbf_reward_penalty}")
    print(f"Obs dim:      {FLATTENED_OBS_SIZE}")
    print(f"Parallel envs:{cfg['env']['num_envs']}, Max iters: {cfg['runner']['max_iterations']}")

    # --- Log Directory ---
    base_log_dir = get_log_dir()
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"Base log directory for TensorBoard logs: {base_log_dir}")

    # --- Environment Setup ---
    num_envs = cfg['env']['num_envs']
    print(f"Run name updated to: {cfg['runner']['run_name']}")
    env_kwargs = {k: v for k, v in cfg['env'].items() if k not in ['env_id', 'num_envs', 'dynamics_model']}

    # Instantiate the selected environment
    vec_env = UnifiedNavigationEnv(
        num_envs=num_envs,
        use_cbf_action_filtering=use_cbf_action_filtering,
        use_cbf_reward_penalty=use_cbf_reward_penalty,
        dynamics_model=dynamics_model,
        **env_kwargs
    )
    
    # add the ability to change run_name to be timestamped
    if not args.headless:
        vec_env.render_mode = "human"
    

    try:
        runner = OnPolicyRunner(
            env=vec_env,
            train_cfg=cfg,
            log_dir=base_log_dir,
            device=cfg['device']
        )
        print("\nOnPolicyRunner initialized successfully.")
        # print("--> Using NaiveNavigationEnv as vectorized environment.") # Removed redundant print
        print("--> Check TensorBoard for rollout and reward statistics.\n")

    except Exception as e:
         print(f"\nError initializing OnPolicyRunner: {e}")
         print("Troubleshooting Tips:")
         print(" - Ensure the configuration structure (config.py) matches rsl-rl expectations.")
         print(" - Verify network input/output dimensions match environment spaces.")
         print(" - Check rsl-rl documentation for Runner initialization and required config fields.")
         print(" - Make sure 'nav_env' is correctly installed and importable.\n")
         return

    print(f"Starting training for {cfg['runner']['max_iterations']} iterations...")
    print(f"*** Monitor TensorBoard logs in the subdirectory created within: {base_log_dir} ***")
    print(f"*** To view: run 'tensorboard --logdir {base_log_dir}' (or point to specific run) ***")
    print("*** Look for tags like 'Loss/policy_loss', 'Loss/value_loss', 'rollout/ep_rew_mean', 'rollout/reward_goal_mean', etc. ***\n")
    try:
        runner.learn(num_learning_iterations=cfg['runner']['max_iterations'],
                     init_at_random_ep_len=True)
        print("\n--- Training finished ---")
    except KeyboardInterrupt:
         print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred during training: {e} ---")
    finally:
        print("Performing cleanup (if any)...")
        print("Cleanup complete.")

if __name__ == '__main__':
    train()