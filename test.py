# test.py
import torch
import numpy as np
import os
import re # Import re for potential pattern matching if needed
import argparse # Import argparse

from config import cfg, get_log_dir, FLATTENED_OBS_SIZE

from nav_env.unified_navigation_env import UnifiedNavigationEnv
# --- IMPORTANT: Import your package to trigger environment registration ---
try:
    import nav_env # Replace with your actual package name
except ImportError:
    print("Error: Could not import the package 'nav_env'.")
    print("Ensure the package is installed and contains the registered NaiveNavigationEnv.")
    exit()
# ------------------------------------------------------------------------

# Attempt to import rsl_rl components
try:
    from rsl_rl.runners import OnPolicyRunner
    # Import the specific policy class used during training
    from rsl_rl.modules import ActorCritic # Or ActorCriticRecurrent, etc.
    _RSL_RL_AVAILABLE = True
except ImportError:
    print("Warning: rsl-rl components not found. Ensure rsl-rl is installed correctly.")
    print("Testing script requires rsl-rl to function.")
    _RSL_RL_AVAILABLE = False

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Test navigation agent policy")
    parser.add_argument('--env', type=str, default='naive', help="Environment type to test (naive, cbf, reward_only, filter_only, etc.)")
    parser.add_argument("--use_cbf_action_filtering", action="store_true", help="Use CBF action filtering")
    parser.add_argument("--use_cbf_reward_penalty", action="store_true", help="Use CBF reward penalty")
    parser.add_argument('--headless', action='store_true',
                        help="Run in headless mode (no GUI)")
    parser.add_argument('--dynamics_model', type=str, default='dynamic',
                        choices=['dynamic', 'quasi_static'],
                        help="Dynamics model: 'dynamic' (double-integrator) or 'quasi_static' (single-integrator)")
    return parser.parse_args()

def find_latest_run_dir(base_log_dir, env_type):
    """Finds the latest timestamped subdirectory for a specific env_type in the base log directory."""
    try:
        # Filter out non-directories and the 'git' directory
        # Also filter based on the environment type in the directory name (e.g., '_naive_' or '_cbf_')
        env_pattern = f"_{env_type}_"
        subdirs = [d for d in os.listdir(base_log_dir)
                   if os.path.isdir(os.path.join(base_log_dir, d)) and d != 'git' and env_pattern in d]
        if not subdirs: return None
        # Sort directories, assuming timestamp format allows chronological sorting
        subdirs.sort()
        return os.path.join(base_log_dir, subdirs[-1])
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error finding latest run directory in {base_log_dir} for env {env_type}: {e}")
        return None

def test():
    if not _RSL_RL_AVAILABLE:
        return

    args = parse_args() # Parse command-line arguments

    print("Starting policy evaluation...")
    print(f"Using device: {cfg['device']}")
    print(f"Environment: {args.env.upper()}NavigationEnv (vectorized)") # Use parsed env name

    # --- Environment Setup ---
    render_mode = "human" if cfg['runner']['render_test'] else None
    env_kwargs = {k: v for k, v in cfg['env'].items() if k not in ['env_id', 'num_envs']}
    # Pass the device to the environment
    eval_env = UnifiedNavigationEnv(
        render_mode=render_mode,
        num_envs=1,
        noise_level=0.0,
        device=cfg['device'],
        use_cbf_action_filtering=args.use_cbf_action_filtering,
        use_cbf_reward_penalty=args.use_cbf_reward_penalty,
        dynamics_model=args.dynamics_model,
        **env_kwargs
    )
    print(f"--> Using UnifiedNavigationEnv for evaluation.")
    print(f"--> use_cbf_action_filtering: {args.use_cbf_action_filtering}")
    print(f"--> use_cbf_reward_penalty: {args.use_cbf_reward_penalty}")

    # Reset returns obs_tensor, extras_dict
    _, extras = eval_env.reset()
    # Access observation dict from extras
    print(f"Evaluation Environment Observation Dict Keys: {extras['observations'].keys()}")
    cfg['runner']['experiment_name'] = f"{cfg['runner']['experiment_name']}_{args.env}"


    # --- Load Policy ---
    base_log_dir = get_log_dir()
    #move one level up to get the base log dir
    base_log_dir = os.path.dirname(base_log_dir)
    if cfg['runner']['load_run'] == -1:
        # Try to find the latest subdirectory for the specific environment type
        latest_run_subdir = find_latest_run_dir(base_log_dir, args.env) # Pass env type
        if latest_run_subdir is not None:
             trained_model_log_dir = latest_run_subdir
             print(f"Found latest run directory for '{args.env}' env: {trained_model_log_dir}")
        else:
             # If no specific env subdirs found, fall back to original behavior (might be less reliable)
             print(f"Warning: No run subdirectories found for '{args.env}' env in {base_log_dir}. Trying generic latest.")
             latest_run_subdir = find_latest_run_dir(base_log_dir, "") # Try finding any latest
             if latest_run_subdir:
                 trained_model_log_dir = latest_run_subdir
                 print(f"Found generic latest run directory: {trained_model_log_dir}")
             else:
                 print(f"Error: No suitable run directory found in {base_log_dir}")
                 eval_env.close()
                 return
    else:
        # Construct path for a specific run number/name (assuming it includes env type if needed)
        # Note: This might need adjustment if 'load_run' doesn't contain env info
        run_name = str(cfg['runner']['load_run'])
        # Heuristic: Check if run_name already contains the env type, otherwise append it based on args
        if f"_{args.env}_" not in run_name:
            print(f"Warning: 'load_run' value '{run_name}' might not correspond to the selected env '{args.env}'.")
            # Consider modifying run_name based on args.env if a pattern exists, or rely on user providing correct name.
        trained_model_log_dir = os.path.join(base_log_dir, run_name)

    print(f"Attempting to load model from checkpoint in: {trained_model_log_dir}")
    if not os.path.isdir(trained_model_log_dir):
         print(f"Error: Log directory not found: {trained_model_log_dir}")
         eval_env.close()
         return

    # --- Determine checkpoint path ---
    cfg['runner']['checkpoint'] = -1 #1500
    if cfg['runner']['checkpoint'] == -1:
        # Find the latest checkpoint file
        try:
            checkpoints = [f for f in os.listdir(trained_model_log_dir) if f.startswith('model_') and f.endswith('.pt')]
            if not checkpoints:
                print(f"Error: No checkpoint files found in {trained_model_log_dir}")
                eval_env.close()
                return
            # Sort by iteration number
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            checkpoint_path = os.path.join(trained_model_log_dir, checkpoints[-1])
        except Exception as e:
            print(f"Error finding latest checkpoint: {e}")
            eval_env.close()
            return
    else:
        checkpoint_path = os.path.join(trained_model_log_dir, f"model_{cfg['runner']['checkpoint']}.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        eval_env.close()
        return
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        # Load the checkpoint directly
        loaded_dict = torch.load(checkpoint_path, map_location=cfg['device'])

        # Recreate the policy architecture based on config
        # Resolve dimensions (assuming env config matches training)
        # Need a temporary way to get obs/priv_obs dims if not stored in checkpoint
        # Let's assume the config file `cfg` has the necessary policy structure info
        temp_obs, temp_extras = eval_env.get_observations() # Use get_observations to get tensor dims
        num_obs = temp_obs.shape[1]
        # Check for privileged obs type based on config/extras
        priv_obs_type = None # Determine this based on training config if possible
        if priv_obs_type and priv_obs_type in temp_extras['observations']:
             num_privileged_obs = temp_extras['observations'][priv_obs_type].shape[1]
        else:
             num_privileged_obs = num_obs # Default if no privileged obs

        # Make a copy to avoid modifying the global cfg dict
        policy_cfg_copy = cfg['policy'].copy()
        # Pop class_name from the copy
        policy_class_name = policy_cfg_copy.pop("class_name", "ActorCritic")
        try:
            policy_class = eval(policy_class_name)
        except NameError:
             # Fallback or specific import if eval fails or is not desired
             if policy_class_name == "ActorCritic":
                 from rsl_rl.modules import ActorCritic
                 policy_class = ActorCritic
             # Add elif for other known policy classes if needed
             else:
                 raise ImportError(f"Could not find or eval policy class: {policy_class_name}")

        policy = policy_class(
            num_obs, num_privileged_obs, eval_env.num_actions, **policy_cfg_copy # Use the copy for kwargs
        ).to(cfg['device'])

        # Load state dict
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval() # Set to evaluation mode
        print("Policy loaded successfully from checkpoint.")

        # Load normalizer if it exists and was used
        obs_normalizer = None
        if cfg['runner'].get('empirical_normalization', False) and 'obs_norm_state_dict' in loaded_dict: # Use .get for safety
            from rsl_rl.modules import EmpiricalNormalization
            # Ensure normalizer is created before loading state dict
            obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(cfg['device'])
            obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
            obs_normalizer.eval() # Set normalizer to eval mode
            print("Observation normalizer loaded.")

    except Exception as e:
        print(f"Error loading policy from checkpoint: {e}")
        eval_env.close()
        return

    # --- Evaluation Loop ---
    total_rewards = []
    total_successes = 0
    num_episodes = cfg['runner']['test_episodes']
    if args.headless:
        eval_env.render_mode = "headless"
    failure_counts = {'obstacle': 0, 'wall': 0, 'timeout': 0}  # <-- add

    for episode in range(num_episodes):
        # Reset returns obs_tensor, extras_dict
        obs_tensor, extras = eval_env.reset(seed=cfg['seed'] + episode)
        # Save initial configuration for potential re-render on failure
        initial_robot_pos = extras.get('robot_position', None)
        initial_goal_pos = extras.get('goal_position', None)
        initial_obstacle_positions = extras.get('obstacle_positions', None)

        # Compute initial h values (if supported by the environment)
        initial_h_robot, initial_h_goal = None, None
        try:
            if hasattr(eval_env, "h_function") and callable(getattr(eval_env, "h_function")):
                obstacle_radii = getattr(eval_env, "_obstacle_radii", None)
                if obstacle_radii is not None and initial_robot_pos is not None and initial_obstacle_positions is not None:
                    initial_h_robot = eval_env.h_function(
                        initial_robot_pos.to(eval_env.device),
                        initial_obstacle_positions.to(eval_env.device),
                        obstacle_radii.to(eval_env.device),
                    )
                    if initial_goal_pos is not None:
                        initial_h_goal = eval_env.h_function(
                            initial_goal_pos.to(eval_env.device),
                            initial_obstacle_positions.to(eval_env.device),
                            obstacle_radii.to(eval_env.device),
                        )
        except Exception as e:
            print(f"Warning: could not compute initial h values: {e}")

        # obs_tensor is already the flattened tensor on the correct device
        done_tensor = torch.zeros(1, dtype=torch.bool, device=cfg['device']) # Use tensor for done flag
        episode_reward = 0.0 # Use float
        step_count = 0

        while not done_tensor.any():
            # Apply normalization if loaded
            if obs_normalizer:
                # Ensure input to normalizer is on the correct device
                obs_tensor = obs_tensor.to(cfg['device'])
                obs_tensor_normalized = obs_normalizer(obs_tensor)
            else:
                obs_tensor_normalized = obs_tensor

            # Ensure the final tensor going into the policy is on the correct device
            obs_tensor_normalized = obs_tensor_normalized.to(cfg['device'])

            with torch.no_grad():
                # Use the loaded policy directly
                # Ensure policy is on the correct device (optional redundancy)
                # policy.to(cfg['device'])
                actions = policy.act_inference(obs_tensor_normalized)
                # action is already a tensor on the correct device

            # Step returns obs_tensor, reward_tensor, done_tensor, extras_dict
            # Ensure obs_tensor from step is on the correct device for the next iteration
            obs_tensor, reward_tensor, done_tensor, extras = eval_env.step(actions)
            # print(extras['log']['reward_h_potential'])
            obs_tensor = obs_tensor.to(cfg['device']) # Ensure device consistency for next loop iteration

            episode_reward += reward_tensor.sum().item()
            step_count += 1

            # Optionally render
            # if render_mode == "human" and not args.headless:
            #     eval_env.render()

        total_rewards.append(episode_reward)
        # Access success flag from the 'log' or 'episode' part of extras, convert tensor to bool
        success_flag = False
        if 'log' in extras and 'success' in extras['log']:
             success_flag = bool(extras['log']['success'].any().item())
        elif 'episode' in extras and 'success' in extras['episode']:
             success_flag = bool(extras['episode']['success'].any().item())

        # --- classify failure type ---
        collided_obstacle_flag, collided_wall_flag = False, False
        if 'log' in extras:
            if 'collided_obstacle' in extras['log']:
                collided_obstacle_flag = bool(extras['log']['collided_obstacle'].any().item())
            if 'collided_wall' in extras['log']:
                collided_wall_flag = bool(extras['log']['collided_wall'].any().item())
        elif 'episode' in extras:
            if 'collided_obstacle' in extras['episode']:
                collided_obstacle_flag = bool(extras['episode']['collided_obstacle'].any().item())
            if 'collided_wall' in extras['episode']:
                collided_wall_flag = bool(extras['episode']['collided_wall'].any().item())

        if success_flag:
            total_successes += 1
        else:
            if collided_obstacle_flag:
                failure_counts['obstacle'] += 1
            elif collided_wall_flag:
                failure_counts['wall'] += 1
            else:
                failure_counts['timeout'] += 1

        print(f"Episode {episode + 1}/{num_episodes} finished in {step_count} steps. Reward: {episode_reward:.2f}. Success: {success_flag}")

        # Print failure reason and re-render the initial configuration before pause
        if not success_flag:
            reason = "obstacle collision" if collided_obstacle_flag else ("wall collision" if collided_wall_flag else "timeout")
            print(f"Failure reason: {reason}")
            # Print initial h values if available
            try:
                if initial_h_robot is not None:
                    h_r = float(initial_h_robot.flatten()[0].item())
                else:
                    h_r = None
                if initial_h_goal is not None:
                    h_g = float(initial_h_goal.flatten()[0].item())
                else:
                    h_g = None
                print(f"Initial h(robot): {h_r if h_r is not None else 'N/A'}, h(goal): {h_g if h_g is not None else 'N/A'}")
            except Exception as e:
                print(f"Warning: could not print initial h values: {e}")

            # Re-render the initial setup (only if human render mode)
            if getattr(eval_env, "render_mode", None) == "human":
                try:
                    if initial_robot_pos is not None: eval_env._robot_pos = initial_robot_pos.clone().to(eval_env.device)
                    if initial_goal_pos is not None: eval_env._goal_pos = initial_goal_pos.clone().to(eval_env.device)
                    if initial_obstacle_positions is not None: eval_env._obstacle_positions = initial_obstacle_positions.clone().to(eval_env.device)
                    if hasattr(eval_env, "_last_velocity"): eval_env._last_velocity[:] = 0
                    if hasattr(eval_env, "_elapsed_steps") and eval_env._elapsed_steps is not None: eval_env._elapsed_steps[:] = 0
                    eval_env.render()
                except Exception as e:
                    print(f"Warning: could not re-render initial setup: {e}")

            # try:
            #     input("Paused: success is 0.0. Press Enter to continue...")
            # except KeyboardInterrupt:
            #     pass

    eval_env.close()

    # --- Report Results ---
    mean_reward = np.mean(total_rewards) if total_rewards else 0.0
    std_reward = np.std(total_rewards) if total_rewards else 0.0
    success_rate = (total_successes / num_episodes) if num_episodes > 0 else 0.0
    fail_total = num_episodes - total_successes
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"results_{args.env}_{timestamp}"
    summary_lines = [
        f"=== Evaluation Summary: {args.env.upper()} ===",
        f"Timestamp:      {timestamp}",
        f"Dynamics:       {args.dynamics_model}",
        f"CBF Filter:     {args.use_cbf_action_filtering}",
        f"CBF Penalty:    {args.use_cbf_reward_penalty}",
        f"Agents:         {cfg['env']['num_agents']}",
        f"Obstacles:      {cfg['env']['num_obstacles']}",
        f"Episodes:       {num_episodes}",
        f"Mean Reward:    {mean_reward:.2f} +/- {std_reward:.2f}",
        f"Success Rate:   {success_rate:.2%}",
        f"Failures:       {fail_total} ({(fail_total / num_episodes):.2%})",
        f" - Obstacle:    {failure_counts['obstacle']} ({(failure_counts['obstacle'] / num_episodes):.2%})",
        f" - Wall:        {failure_counts['wall']} ({(failure_counts['wall'] / num_episodes):.2%})",
        f" - Timeout:     {failure_counts['timeout']} ({(failure_counts['timeout'] / num_episodes):.2%})",
        "=" * 40,
    ]

    print("\n" + "\n".join(summary_lines))

    result_dir = os.path.join("results", f"results_{args.env}", run_name)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{run_name}.txt")
    with open(result_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Results saved to {result_path}")


if __name__ == '__main__':
    # --- Configuration for Testing ---
    # These might be overridden by finding the latest run/checkpoint
    cfg['runner']['load_run'] = -1 # Load latest run directory (logic now considers env type)
    cfg['runner']['checkpoint'] = -1 # Load latest checkpoint from that directory
    cfg['runner']['render_test'] = True # Set True to watch the agent play

    test() # test function now uses parsed args