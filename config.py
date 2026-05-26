# config.py
import os
import torch
import yaml

# ─── Load YAML ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "nav_config.yaml")

with open(_CONFIG_PATH) as _f:
    _y = yaml.safe_load(_f)

# Convenience aliases into YAML sections
_env  = _y["environment"]
_dyn  = _y["dynamics"]
_cbf  = _y["cbf"]
_pol  = _y["policy"]
_alg  = _y["algorithm"]
_tr   = _y["training"]
_te   = _y["testing"]

# ─── Derived constants (used by train.py / test.py directly) ──────────────────
NUM_AGENTS    = int(_env["num_agents"])
NUM_OBSTACLES = int(_env["num_obstacles"])
WORLD_SIZE    = float(_env["world_size"])

# Observation dimension (must match _get_obs() key order — sorted alphabetically):
#   goal_pos(2) + obstacles(3 if any) + other_agents((A-1)*3 if A>1) + robot_pos(2) + robot_vel(2)
FLATTENED_OBS_SIZE = (
    2                                              # goal_pos
    + (3 if NUM_OBSTACLES > 0 else 0)             # obstacles (closest only)
    + ((NUM_AGENTS - 1) * 3 if NUM_AGENTS > 1 else 0)  # other_agents
    + 2                                            # robot_pos
    + 2                                            # robot_vel
)

# ─── Unified cfg dict (interface consumed by rsl_rl OnPolicyRunner) ───────────
cfg = {
    "seed":   int(_tr["seed"]),
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "env": {
        "env_id":            "NaiveNavigation-v0",
        "num_envs":          int(_tr["num_envs"]),
        "world_size":        float(_env["world_size"]),
        "num_obstacles":     NUM_OBSTACLES,
        "num_agents":        NUM_AGENTS,
        "robot_radius":      float(_env["robot_radius"]),
        "obstacle_radius":   float(_env["obstacle_radius"]),
        "goal_radius":       float(_env["goal_radius"]),
        "max_velocity":      float(_env["max_velocity"]),
        "max_acceleration":  float(_env["max_acceleration"]),
        "dt":                float(_env["dt"]),
        "max_episode_steps": int(_env["max_episode_steps"]),
        "noise_level":       float(_env["noise_level"]),
        "dynamics_model":    str(_dyn["model"]),
    },

    "policy": {
        "class_name":          "ActorCritic",
        "actor_hidden_dims":   list(_pol["actor_hidden_dims"]),
        "critic_hidden_dims":  list(_pol["critic_hidden_dims"]),
        "activation":          str(_pol["activation"]),
    },

    "algorithm": {
        "class_name":           "PPO",
        "gamma":                float(_alg["gamma"]),
        "lam":                  float(_alg["lam"]),
        "learning_rate":        float(_alg["learning_rate"]),
        "entropy_coef":         float(_alg["entropy_coef"]),
        "clip_param":           float(_alg["clip_param"]),
        "num_learning_epochs":  int(_alg["num_learning_epochs"]),
        "num_mini_batches":     int(_alg["num_mini_batches"]),
    },

    "runner": {
        "run_name":          str(_tr["run_name"]),
        "experiment_name":   str(_tr["experiment_name"]),
        "max_iterations":    int(_tr["max_iterations"]),
        "num_steps_per_env": int(_tr["num_steps_per_env"]),
        "save_interval":     int(_tr["save_interval"]),
        "log_interval":      int(_tr["log_interval"]),
        "resume":            False,
        "load_run":          -1,
        "checkpoint":        -1,
        "test_episodes":     int(_te["test_episodes"]),
        "render_test":       bool(_te["render"]),
        "empirical_normalization":  False,
        "log_reward_components":    True,
    },

    # CBF defaults from YAML (train.py / test.py CLI flags override these)
    "cbf": {
        "use_action_filtering": bool(_cbf["use_action_filtering"]),
        "use_reward_penalty":   bool(_cbf["use_reward_penalty"]),
    },
}

# Flatten runner keys to top level (rsl_rl OnPolicyRunner expects them there)
cfg.update(cfg["runner"])


# ─── Log directory helper ─────────────────────────────────────────────────────
def get_log_dir(base_log_path="logs"):
    return os.path.join(
        base_log_path,
        cfg["runner"]["experiment_name"],
        cfg["runner"]["run_name"],
    )
