import os
import json
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
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train navigation agent with rsl-rl")
    parser.add_argument('--env', type=str, default='naive', help="Environment type to use (naive, cbf, reward_only, filter_only, etc.)")
    parser.add_argument("--use_cbf_action_filtering", action="store_true", help="Use CBF action filtering")
    parser.add_argument("--use_cbf_reward_penalty", action="store_true", help="Use CBF reward penalty")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (no GUI).")
    parser.add_argument('--dynamics_model', type=str, default='quasi_static',
                        choices=['dynamic', 'quasi_static'],
                        help=("Dynamics model: 'quasi_static' (single-integrator, original "
                              "replication default) or 'dynamic' (double-integrator / acceleration)."))
    parser.add_argument('--obs_layout', type=str, default='legacy',
                        choices=list(UnifiedNavigationEnv.OBS_LAYOUTS),
                        help=("Observation key layout. 'legacy' uses 'last_velocity' "
                              "(original replication). 'current' uses 'robot_vel'."))
    parser.add_argument('--num_agents', type=int, default=None,
                        help="Number of agents per world (default: config NUM_AGENTS=1)")
    parser.add_argument('--num_obstacles', type=int, default=None,
                        help="Number of static obstacles per world (default: config NUM_OBSTACLES=3)")
    return parser.parse_args()

def flattened_obs_size(num_obstacles: int, num_agents: int) -> int:
    """Match UnifiedNavigationEnv flat obs: goal(2) + closest obs(3*O) + pos(2) + vel(2) + other agents."""
    return 2 + (num_obstacles * 3) + 2 + 2 + (num_agents - 1) * 3

def find_run_dir_with_checkpoints(base_log_dir, run_name):
    """Return the directory that contains model checkpoints for this run."""
    candidates = [base_log_dir]
    nested = os.path.join(base_log_dir, run_name)
    if nested != base_log_dir:
        candidates.append(nested)
    for path in candidates:
        if os.path.isdir(path) and any(
            f.startswith("model_") and f.endswith(".pt") for f in os.listdir(path)
        ):
            return path
    return base_log_dir

def write_env_meta(run_dir, meta):
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "env_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"--> Wrote env metadata to {path}")

def train():
    """Initializes and runs the rsl-rl training process."""
    if not _RSL_RL_AVAILABLE:
        print("Cannot proceed without rsl-rl installed.")
        return

    args = parse_args() # Parse arguments
    num_agents = args.num_agents if args.num_agents is not None else cfg['env']['num_agents']
    num_obstacles = (
        args.num_obstacles if args.num_obstacles is not None else cfg['env']['num_obstacles']
    )

    # Add environment type and timestamp to run name and experiment name for better tracking
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    
    cfg['runner']['run_name'] = f"{cfg['runner']['run_name']}_{args.env}_{timestamp}"
    cfg['runner']['experiment_name'] = f"{cfg['runner']['experiment_name']}_{args.env}"

    print("--- Starting Training ---")
    print(f"Run Name: {cfg['runner']['run_name']}")
    print(f"Experiment Name: {cfg['runner']['experiment_name']}")
    print(f"Device: {cfg['device']}")
    print(f"Environment: {args.env.upper()}NavigationEnv (vectorized)") # Use parsed env name
    print(f"Observation Size (Flattened): {flattened_obs_size(num_obstacles, num_agents)}")
    print(f"Number of Parallel Environments: {cfg['env']['num_envs']}")
    print(f"Max Training Iterations: {cfg['runner']['max_iterations']}")

    # --- Log Directory ---
    base_log_dir = get_log_dir()
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"Base log directory for TensorBoard logs: {base_log_dir}")

    # --- Environment Setup ---
    num_envs = cfg['env']['num_envs']
    print(f"Run name updated to: {cfg['runner']['run_name']}")
    env_kwargs = {k: v for k, v in cfg['env'].items() if k not in ['env_id', 'num_envs', 'num_agents']}
    env_kwargs['num_obstacles'] = num_obstacles

    # Instantiate the selected environment
    vec_env = UnifiedNavigationEnv(
        num_envs=num_envs,
        num_agents=num_agents,
        noise_level=0.0,
        use_cbf_action_filtering=args.use_cbf_action_filtering,
        use_cbf_reward_penalty=args.use_cbf_reward_penalty,
        dynamics_model=args.dynamics_model,
        obs_layout=args.obs_layout,
        **env_kwargs
    )
    print(f"--> Using UnifiedNavigationEnv as vectorized environment.")
    print(f"--> use_cbf_action_filtering: {args.use_cbf_action_filtering}")
    print(f"--> use_cbf_reward_penalty: {args.use_cbf_reward_penalty}")
    print(f"--> dynamics_model: {args.dynamics_model}")
    print(f"--> obs_layout: {args.obs_layout}")
    print(f"--> num_agents: {num_agents}")
    print(f"--> num_obstacles: {num_obstacles}")
    
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
        print("--> Check TensorBoard for rollout and reward statistics.\n")

        run_dir = find_run_dir_with_checkpoints(base_log_dir, cfg['runner']['run_name'])
        meta = {
            "env": args.env,
            "obs_layout": args.obs_layout,
            "dynamics_model": args.dynamics_model,
            "num_agents": num_agents,
            "num_obstacles": num_obstacles,
            "use_cbf_action_filtering": bool(args.use_cbf_action_filtering),
            "use_cbf_reward_penalty": bool(args.use_cbf_reward_penalty),
        }
        try:
            write_env_meta(run_dir, meta)
        except Exception as e:
            print(f"Warning: failed to write env_meta.json: {e}")

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
