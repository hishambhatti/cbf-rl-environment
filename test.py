# test.py
import torch
import numpy as np
import os
import re # Import re for potential pattern matching if needed
import json
import argparse # Import argparse
from datetime import datetime

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
    parser.add_argument('--dynamics_model', type=str, default=None,
                        choices=['dynamic', 'quasi_static'],
                        help=("Dynamics model. If unset, auto-detected from env_meta.json "
                              "in the run directory; falls back to 'quasi_static' (original "
                              "replication default)."))
    parser.add_argument('--obs_layout', type=str, default=None,
                        choices=list(UnifiedNavigationEnv.OBS_LAYOUTS),
                        help=("Observation key layout. If unset, auto-detected from "
                              "env_meta.json; falls back to 'legacy' (last_velocity)."))
    parser.add_argument('--num_agents', type=int, default=None,
                        help="Number of agents per world (default: config NUM_AGENTS=1)")
    parser.add_argument(
        '--load_run', type=str, default=None,
        help=("Training run directory name (e.g. navigation_cbf_20260529_172331) or "
              "full path under logs/navigation_<env>/. Default: latest run for --env."),
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help=("Checkpoint to evaluate: iteration number (e.g. 976), "
              "model_976.pt, or path to a .pt file. Default: latest in run dir."),
    )
    parser.add_argument(
        '--save_video_interval', type=int, default=100,
        help=("Save trajectory PNG + MP4 every N episodes (e.g. 50, 100, ...). "
              "Set to 0 to disable. Works with or without --headless. Default: 100."),
    )
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

def find_env_meta_path(run_dir):
    """Locate env_meta.json next to checkpoints (handles legacy double-nested runs)."""
    direct = os.path.join(run_dir, "env_meta.json")
    if os.path.isfile(direct):
        return direct
    nested = os.path.join(run_dir, os.path.basename(run_dir), "env_meta.json")
    if os.path.isfile(nested):
        return nested
    return direct

def resolve_run_dir(base_log_dir, env_type, load_run_arg):
    """Resolve training run directory from CLI arg or latest match."""
    if load_run_arg is not None:
        if os.path.isdir(load_run_arg):
            return os.path.abspath(load_run_arg)
        candidate = os.path.join(base_log_dir, load_run_arg)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(
            f"Training run directory not found: {load_run_arg} (also tried {candidate})"
        )

    latest_run_subdir = find_latest_run_dir(base_log_dir, env_type)
    if latest_run_subdir is not None:
        return latest_run_subdir

    latest_run_subdir = find_latest_run_dir(base_log_dir, "")
    if latest_run_subdir is not None:
        return latest_run_subdir

    raise FileNotFoundError(f"No suitable run directory found in {base_log_dir}")

def resolve_checkpoint_path(run_dir, checkpoint_arg):
    """Resolve checkpoint file from CLI arg or latest model_*.pt in run_dir."""
    if checkpoint_arg is None:
        checkpoints = [
            f for f in os.listdir(run_dir)
            if f.startswith("model_") and f.endswith(".pt")
        ]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {run_dir}")
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        return os.path.join(run_dir, checkpoints[-1])

    if os.path.isfile(checkpoint_arg):
        return os.path.abspath(checkpoint_arg)

    candidate = os.path.join(run_dir, checkpoint_arg)
    if os.path.isfile(candidate):
        return candidate

    if checkpoint_arg.startswith("model_") and checkpoint_arg.endswith(".pt"):
        raise FileNotFoundError(f"Checkpoint file not found: {candidate}")

    iteration = int(checkpoint_arg)
    path = os.path.join(run_dir, f"model_{iteration}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path

def episode_outcome_label(success_flag, collided_obstacle_flag, collided_wall_flag):
    if success_flag:
        return "success"
    if collided_obstacle_flag:
        return "obstacle_collision"
    if collided_wall_flag:
        return "wall_collision"
    return "timeout"

def _capture_episode_layout(eval_env):
    """Freeze world-0 layout at episode start for trajectory rendering."""
    return {
        "goal_pos": eval_env._goal_pos[0].detach().cpu().numpy().copy(),
        "obstacle_pos": eval_env._obstacle_positions[0].detach().cpu().numpy().copy(),
        "obstacle_radii": eval_env._obstacle_radii[0].detach().cpu().numpy().copy(),
        "world_size": eval_env.world_size,
        "robot_radius": eval_env.robot_radius,
        "goal_radius": eval_env.goal_radius,
    }

def _terminal_robot_pos(extras, eval_env):
    """Robot positions at episode end, before the env auto-resets."""
    terminal = extras.get("terminal_layout")
    if terminal is not None:
        return terminal["robot_pos"][0].detach().cpu().numpy()
    return eval_env._robot_pos[0].detach().cpu().numpy()

def _draw_episode_scene(ax, episode_layout, trajectories, step_idx, outcome_label):
    """Draw obstacles, goals, dashed trajectories, and robots at step_idx."""
    import matplotlib.patches as patches

    goal_pos = episode_layout["goal_pos"]
    obstacle_pos = episode_layout["obstacle_pos"]
    obstacle_radii = episode_layout["obstacle_radii"]
    world_size = episode_layout["world_size"]
    robot_radius = episode_layout["robot_radius"]
    goal_radius = episode_layout["goal_radius"]
    ax.set_xlim(0, world_size)
    ax.set_ylim(0, world_size)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Episode trajectory ({outcome_label.replace('_', ' ')})")

    for i, (ox, oy) in enumerate(obstacle_pos):
        radius = float(obstacle_radii[i])
        ax.add_patch(patches.Circle((ox, oy), radius, fc="red", alpha=0.6))

    agent_colors = ["blue", "royalblue", "deepskyblue", "steelblue"]
    goal_colors = ["green", "limegreen", "forestgreen", "darkgreen"]
    for agent_idx, traj in enumerate(trajectories):
        traj_np = np.asarray(traj, dtype=np.float32)
        if len(traj_np) == 0:
            continue
        color = agent_colors[agent_idx % len(agent_colors)]
        goal_color = goal_colors[agent_idx % len(goal_colors)]
        ax.plot(
            traj_np[:, 0], traj_np[:, 1], "--", color=color,
            alpha=0.8, linewidth=1.5, label=f"Agent {agent_idx} path",
        )
        ax.plot(traj_np[0, 0], traj_np[0, 1], "o", color=color, markersize=8, label=f"Agent {agent_idx} start")
        end_idx = min(step_idx, len(traj_np) - 1)
        ax.plot(traj_np[end_idx, 0], traj_np[end_idx, 1], "x", color=color, markersize=10)
        ax.add_patch(patches.Circle(
            (goal_pos[agent_idx, 0], goal_pos[agent_idx, 1]),
            goal_radius, fc=goal_color, alpha=0.8,
        ))
        ax.add_patch(patches.Circle(
            (traj_np[end_idx, 0], traj_np[end_idx, 1]),
            robot_radius, fc=color, alpha=0.8,
        ))

    ax.legend(loc="upper right", fontsize=8)

def _fig_to_rgb(fig):
    """Return an RGB uint8 array from a matplotlib figure (mpl 3.10+ compatible)."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()

def render_episode_frame(episode_layout, trajectories, step_idx, outcome_label):
    """Render one animation frame and return an RGB uint8 array."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    _draw_episode_scene(ax, episode_layout, trajectories, step_idx, outcome_label)
    frame = _fig_to_rgb(fig)
    plt.close(fig)
    return frame

def save_trajectory_image(episode_layout, trajectories, outcome_label, output_path):
    """Save a static PNG with the full dashed trajectory."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    final_step = max((len(traj) for traj in trajectories), default=1) - 1
    _draw_episode_scene(ax, episode_layout, trajectories, final_step, outcome_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

def _flush_file(path):
    """Ensure a file is visible on disk (helps on NFS and after interrupt)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass

def _video_frame_indices(num_steps, max_frames=150):
    """Subsample trajectory indices for MP4 (timeouts can be 1000+ steps)."""
    if num_steps <= max_frames:
        return list(range(num_steps))
    indices = np.linspace(0, num_steps - 1, max_frames, dtype=int)
    return sorted(set(indices.tolist()))

def _pad_frames_for_h264(frames, block=16):
    """Pad frames so width/height are divisible by 16 (avoids ffmpeg resize warnings)."""
    padded_frames = []
    for frame in frames:
        h, w = frame.shape[:2]
        nh = ((h + block - 1) // block) * block
        nw = ((w + block - 1) // block) * block
        if nh == h and nw == w:
            padded_frames.append(frame)
            continue
        out = np.zeros((nh, nw, 3), dtype=np.uint8)
        out[:h, :w] = frame
        padded_frames.append(out)
    return padded_frames

def save_mp4(frames, mp4_path, fps=20):
    """Save RGB frames to MP4 via matplotlib+ffmpeg, or imageio if available."""
    frames = _pad_frames_for_h264(frames)
    errors = []

    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        im = ax.imshow(frames[0])

        def _update(i):
            im.set_array(frames[i])
            return [im]

        anim = FuncAnimation(fig, _update, frames=len(frames), blit=True)
        anim.save(mp4_path, writer=FFMpegWriter(fps=fps))
        plt.close(fig)
        return
    except Exception as e:
        errors.append(str(e))

    try:
        import imageio
        imageio.mimsave(mp4_path, frames, fps=fps)
        return
    except ImportError:
        errors.append("imageio not installed")
    except Exception as e:
        errors.append(f"imageio: {e}")

    raise RuntimeError(
        "Could not write MP4 (need ffmpeg on PATH, or pip install imageio imageio-ffmpeg). "
        + "; ".join(errors)
    )

def save_episode_media(episode_layout, trajectories, outcome_label, base_path, fps=20, max_video_frames=150):
    """Save trajectory PNG and MP4 for one recorded episode."""
    if not trajectories or all(len(traj) == 0 for traj in trajectories):
        return

    num_steps = max(len(traj) for traj in trajectories)
    png_path = f"{base_path}.png"
    mp4_path = f"{base_path}.mp4"

    # PNG first so it appears on disk immediately (before slow video rendering).
    save_trajectory_image(episode_layout, trajectories, outcome_label, png_path)
    _flush_file(png_path)
    print(f"  Saved PNG: {png_path}", flush=True)

    frame_indices = _video_frame_indices(num_steps, max_frames=max_video_frames)
    if len(frame_indices) < num_steps:
        print(
            f"  Rendering MP4 with {len(frame_indices)}/{num_steps} subsampled frames...",
            flush=True,
        )
    else:
        print(f"  Rendering MP4 with {len(frame_indices)} frames...", flush=True)

    frames = [
        render_episode_frame(episode_layout, trajectories, step_idx, outcome_label)
        for step_idx in frame_indices
    ]

    try:
        save_mp4(frames, mp4_path, fps=fps)
        _flush_file(mp4_path)
        print(f"  Saved MP4: {mp4_path}", flush=True)
    except Exception as e:
        print(f"  Warning: could not save MP4 to {mp4_path} ({e}). PNG is at {png_path}.", flush=True)

def test():
    if not _RSL_RL_AVAILABLE:
        return

    args = parse_args() # Parse command-line arguments

    print("Starting policy evaluation...")
    print(f"Using device: {cfg['device']}")
    print(f"Environment: {args.env.upper()}NavigationEnv (vectorized)") # Use parsed env name

    cfg['runner']['experiment_name'] = f"{cfg['runner']['experiment_name']}_{args.env}"

    # --- Locate run directory first so we can read env_meta.json before building env
    base_log_dir = get_log_dir()
    base_log_dir = os.path.dirname(base_log_dir)
    try:
        trained_model_log_dir = resolve_run_dir(base_log_dir, args.env, args.load_run)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    print(f"Using training run directory: {trained_model_log_dir}")

    meta_path = find_env_meta_path(trained_model_log_dir)
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            print(f"Loaded env metadata: {meta}")
        except Exception as e:
            print(f"Warning: could not read {meta_path}: {e}")

    if args.obs_layout is not None:
        resolved_obs_layout = args.obs_layout
        layout_source = "CLI flag --obs_layout"
    elif meta.get("obs_layout") in UnifiedNavigationEnv.OBS_LAYOUTS:
        resolved_obs_layout = meta["obs_layout"]
        layout_source = f"env_meta.json ({meta_path})"
    else:
        resolved_obs_layout = "legacy"
        layout_source = "default (legacy, original replication)"
    print(f"--> obs_layout resolved to '{resolved_obs_layout}' from {layout_source}")

    if args.dynamics_model is not None:
        resolved_dynamics_model = args.dynamics_model
        dyn_source = "CLI flag --dynamics_model"
    elif meta.get("dynamics_model") in ("dynamic", "quasi_static"):
        resolved_dynamics_model = meta["dynamics_model"]
        dyn_source = f"env_meta.json ({meta_path})"
    else:
        resolved_dynamics_model = "quasi_static"
        dyn_source = "default (quasi_static, original replication)"
    print(f"--> dynamics_model resolved to '{resolved_dynamics_model}' from {dyn_source}")

    if args.num_agents is not None:
        resolved_num_agents = args.num_agents
    elif isinstance(meta.get("num_agents"), int) and meta["num_agents"] >= 1:
        resolved_num_agents = meta["num_agents"]
    else:
        resolved_num_agents = cfg['env']['num_agents']
    print(f"--> num_agents resolved to {resolved_num_agents}")

    # --- Environment Setup ---
    render_mode = None if args.headless else ("human" if cfg['runner']['render_test'] else None)
    env_kwargs = {k: v for k, v in cfg['env'].items() if k not in ['env_id', 'num_envs', 'num_agents']}
    eval_env = UnifiedNavigationEnv(
        render_mode=render_mode,
        num_envs=1,
        num_agents=resolved_num_agents,
        noise_level=0.0,
        device=cfg['device'],
        use_cbf_action_filtering=args.use_cbf_action_filtering,
        use_cbf_reward_penalty=args.use_cbf_reward_penalty,
        dynamics_model=resolved_dynamics_model,
        obs_layout=resolved_obs_layout,
        **env_kwargs
    )
    print(f"--> Using UnifiedNavigationEnv for evaluation.")
    print(f"--> use_cbf_action_filtering: {args.use_cbf_action_filtering}")
    print(f"--> use_cbf_reward_penalty: {args.use_cbf_reward_penalty}")

    _, extras = eval_env.reset()
    print(f"Evaluation Environment Observation Dict Keys: {extras['observations'].keys()}")

    # --- Determine checkpoint path ---
    try:
        checkpoint_path = resolve_checkpoint_path(trained_model_log_dir, args.checkpoint)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        eval_env.close()
        return

    checkpoint_name = os.path.basename(checkpoint_path)
    training_run_name = os.path.basename(trained_model_log_dir)
    checkpoint_stem = os.path.splitext(checkpoint_name)[0]
    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        loaded_dict = torch.load(checkpoint_path, map_location=cfg['device'])

        temp_obs, temp_extras = eval_env.get_observations()
        num_obs = temp_obs.shape[1]
        priv_obs_type = None
        if priv_obs_type and priv_obs_type in temp_extras['observations']:
             num_privileged_obs = temp_extras['observations'][priv_obs_type].shape[1]
        else:
             num_privileged_obs = num_obs

        policy_cfg_copy = cfg['policy'].copy()
        policy_class_name = policy_cfg_copy.pop("class_name", "ActorCritic")
        try:
            policy_class = eval(policy_class_name)
        except NameError:
             if policy_class_name == "ActorCritic":
                 from rsl_rl.modules import ActorCritic
                 policy_class = ActorCritic
             else:
                 raise ImportError(f"Could not find or eval policy class: {policy_class_name}")

        policy = policy_class(
            num_obs, num_privileged_obs, eval_env.num_actions, **policy_cfg_copy
        ).to(cfg['device'])

        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()
        print("Policy loaded successfully from checkpoint.")

        obs_normalizer = None
        if cfg['runner'].get('empirical_normalization', False) and 'obs_norm_state_dict' in loaded_dict:
            from rsl_rl.modules import EmpiricalNormalization
            obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(cfg['device'])
            obs_normalizer.load_state_dict(loaded_dict['obs_norm_state_dict'])
            obs_normalizer.eval()
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
        eval_env.render_mode = None
    failure_counts = {'obstacle': 0, 'wall': 0, 'timeout': 0}
    save_video_interval = args.save_video_interval
    record_episode_media = save_video_interval > 0
    if record_episode_media:
        print(f"Saving episode PNG + MP4 every {save_video_interval} episodes.")

    eval_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_run_name = (
        f"results_{args.env}_{training_run_name}_{checkpoint_stem}_{eval_timestamp}"
    )
    result_dir = os.path.join("results", f"results_{args.env}", result_run_name)
    media_dir = os.path.join(result_dir, "episodes")
    if record_episode_media:
        os.makedirs(media_dir, exist_ok=True)
        print(f"Episode media directory: {media_dir}", flush=True)

    for episode in range(num_episodes):
        obs_tensor, extras = eval_env.reset(seed=cfg['seed'] + episode)
        initial_robot_pos = extras.get('robot_position', None)
        initial_goal_pos = extras.get('goal_position', None)
        initial_obstacle_positions = extras.get('obstacle_positions', None)
        should_record = record_episode_media and ((episode + 1) % save_video_interval == 0)
        trajectories = None
        episode_layout = None
        if should_record:
            num_agents = eval_env.num_agents
            episode_layout = _capture_episode_layout(eval_env)
            trajectories = [[] for _ in range(num_agents)]
            robot_start = eval_env._robot_pos[0].detach().cpu().numpy()
            for agent_idx in range(num_agents):
                trajectories[agent_idx].append(robot_start[agent_idx].copy())

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

        done_tensor = torch.zeros(eval_env.num_envs, dtype=torch.bool, device=cfg['device'])
        episode_reward = 0.0
        step_count = 0

        while not done_tensor.all():
            if obs_normalizer:
                obs_tensor = obs_tensor.to(cfg['device'])
                obs_tensor_normalized = obs_normalizer(obs_tensor)
            else:
                obs_tensor_normalized = obs_tensor

            obs_tensor_normalized = obs_tensor_normalized.to(cfg['device'])

            with torch.no_grad():
                actions = policy.act_inference(obs_tensor_normalized)

            obs_tensor, reward_tensor, done_tensor, extras = eval_env.step(actions)
            obs_tensor = obs_tensor.to(cfg['device'])

            episode_reward += reward_tensor.sum().item()
            step_count += 1

            if should_record and trajectories is not None:
                if done_tensor.any():
                    robot_step = _terminal_robot_pos(extras, eval_env)
                else:
                    robot_step = eval_env._robot_pos[0].detach().cpu().numpy()
                for agent_idx in range(eval_env.num_agents):
                    trajectories[agent_idx].append(robot_step[agent_idx].copy())
                if eval_env.render_mode == "human":
                    eval_env.render()

        total_rewards.append(episode_reward)
        success_flag = False
        if 'log' in extras and 'success' in extras['log']:
             success_flag = bool(extras['log']['success'].any().item())
        elif 'episode' in extras and 'success' in extras['episode']:
             success_flag = bool(extras['episode']['success'].any().item())

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

        outcome_label = episode_outcome_label(
            success_flag, collided_obstacle_flag, collided_wall_flag
        )

        print(
            f"Episode {episode + 1}/{num_episodes} finished in {step_count} steps. "
            f"Reward: {episode_reward:.2f}. Success: {success_flag}",
            flush=True,
        )

        if not success_flag:
            reason = "obstacle collision" if collided_obstacle_flag else ("wall collision" if collided_wall_flag else "timeout")
            print(f"Failure reason: {reason}", flush=True)
            try:
                if initial_h_robot is not None:
                    h_r = float(initial_h_robot.flatten()[0].item())
                else:
                    h_r = None
                if initial_h_goal is not None:
                    h_g = float(initial_h_goal.flatten()[0].item())
                else:
                    h_g = None
                print(f"Initial h(robot): {h_r if h_r is not None else 'N/A'}, h(goal): {h_g if h_g is not None else 'N/A'}", flush=True)
            except Exception as e:
                print(f"Warning: could not print initial h values: {e}", flush=True)

        if should_record and trajectories is not None and episode_layout is not None:
            media_base = os.path.join(
                media_dir,
                f"{training_run_name}_{episode + 1:04d}_{outcome_label}",
            )
            print(f"Saving media for episode {episode + 1}...", flush=True)
            save_episode_media(episode_layout, trajectories, outcome_label, media_base)

        if not success_flag and getattr(eval_env, "render_mode", None) == "human":
            try:
                P, A = eval_env._num_parallel_envs, eval_env.num_agents
                O = eval_env.num_obstacles
                if initial_robot_pos is not None:
                    eval_env._robot_pos = (
                        initial_robot_pos.reshape(P, A, 2).clone().to(eval_env.device)
                    )
                if initial_goal_pos is not None:
                    eval_env._goal_pos = (
                        initial_goal_pos.reshape(P, A, 2).clone().to(eval_env.device)
                    )
                if initial_obstacle_positions is not None:
                    eval_env._obstacle_positions = (
                        initial_obstacle_positions.reshape(P, A, O, 2)[:, 0, :, :]
                        .clone().to(eval_env.device)
                    )
                if hasattr(eval_env, "_robot_vel") and eval_env._robot_vel is not None:
                    eval_env._robot_vel[:] = 0
                if hasattr(eval_env, "_elapsed_steps") and eval_env._elapsed_steps is not None:
                    eval_env._elapsed_steps[:] = 0
                eval_env.render()
            except Exception as e:
                print(f"Warning: could not re-render initial setup: {e}", flush=True)

    eval_env.close()

    mean_reward = np.mean(total_rewards) if total_rewards else 0.0
    std_reward = np.std(total_rewards) if total_rewards else 0.0
    success_rate = (total_successes / num_episodes) if num_episodes > 0 else 0.0
    fail_total = num_episodes - total_successes
    summary_lines = [
        f"=== Evaluation Summary: {args.env.upper()} ===",
        f"Timestamp:      {eval_timestamp}",
        f"Training run:   {training_run_name}",
        f"Checkpoint:     {checkpoint_name}",
        f"Training path:  {trained_model_log_dir}",
        f"Dynamics:       {resolved_dynamics_model}",
        f"Obs layout:     {resolved_obs_layout}",
        f"CBF Filter:     {args.use_cbf_action_filtering}",
        f"CBF Penalty:    {args.use_cbf_reward_penalty}",
        f"Agents:         {resolved_num_agents}",
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

    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{result_run_name}.txt")
    with open(result_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Results saved to {result_path}")
    if record_episode_media:
        print(f"Episode media saved under {media_dir}")


if __name__ == '__main__':
    cfg['runner']['render_test'] = True
    test()
