# naive_navigation_env.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Dict, Any, Tuple, Union, List
from rsl_rl.env.vec_env import VecEnv  # Import VecEnv base class
import torch


class UnifiedNavigationEnv(VecEnv):
    """
    Custom vectorized environment for 2D robot navigation with single integrator dynamics.
    Uses PyTorch tensors for internal state representation.

    The agent controls a point robot by setting its velocity [vx, vy].
    The goal is to navigate to a target location while avoiding static circular obstacles.

    **Action Space:** Box(2,) - Velocity command [vx, vy] (handled as tensor)
    **Observation Space (Dict of Tensors):**
        - 'robot_pos': Tensor(num_envs, 2) - Robot's current [x, y] position.
        - 'obstacles': Tensor(num_envs, num_obstacles * 3) - Concatenated [x, y, radius] for each obstacle.
        - 'goal_pos': Tensor(num_envs, 2) - Goal's target [x, y] position.
        - 'last_velocity' or 'robot_vel': Tensor(num_envs, 2) - Robot velocity [vx, vy].
          Key name is controlled by ``obs_layout`` (see below). The flat observation
          vector is built by sorting dict keys alphabetically, so the key name affects
          feature ordering. Use ``obs_layout='legacy'`` for checkpoints trained on the
          original single-agent replication (key ``last_velocity``).

    **Control-input noise:** ``control_noise`` ("low"/"medium"/"high") injects additive
    Gaussian noise into the executed control (velocity for ``quasi_static``,
    acceleration for ``dynamic``) for stochastic-robustness ablation studies. The
    noise std is a fraction of the max control magnitude (see ``CONTROL_NOISE_STD``);
    "low" reproduces the env's legacy default noise level.
    **Reward:** Combination of goal achievement bonus, collision penalty, and distance shaping (tensor).
    **Termination:** Episode ends if the robot reaches the goal or collides with an obstacle (tensor).
    **Truncation:** Can be handled by wrappers (e.g., TimeLimit).
    """

    OBS_LAYOUTS = ("legacy", "current")
    _OBS_VEL_KEY = {"legacy": "last_velocity", "current": "robot_vel"}

    # Stochastic noise injected into the control input (velocity command for
    # quasi_static, acceleration command for dynamic). Used for ablation studies.
    # The value is the std of the additive Gaussian noise expressed as a fraction
    # of the max control magnitude (max_velocity or max_acceleration). "low" is the
    # env's legacy default noise level.
    CONTROL_NOISE_LEVELS = ("low", "medium", "high")
    CONTROL_NOISE_STD = {"low": 0.1, "medium": 0.3, "high": 0.5}

    # metadata and gym.Env inheritance removed

    def __init__(
        self,
        render_mode: Optional[str] = None,
        world_size: float = 10.0,
        num_obstacles: int = 3,
        robot_radius: float = 0.2,
        obstacle_radius: float = 0.5,
        goal_radius: float = 0.3,
        max_velocity: float = 1.0,
        max_acceleration: float = 2.0,
        dt: float = 0.1,
        max_episode_steps: Optional[int] = None,
        num_envs: int = 1,
        num_actions: int = 2,
        device: str = "cpu",  # Add device argument
        use_cbf_action_filtering: bool = True,
        use_cbf_reward_penalty: bool = True,
        noise_level: float = 0.1,
        dynamics_model: str = "quasi_static",  # "dynamic" or "quasi_static"
        num_agents: int = 1,
        obs_layout: str = "legacy",  # "legacy" (last_velocity) or "current" (robot_vel)
        control_noise: str = "low",  # "low" (legacy), "medium" or "high" (Gaussian control-input noise)
    ):
        """
        Initializes the UnifiedNavigationEnv.

        Args:
            render_mode: The rendering mode ('human' or 'rgb_array').
            world_size: The size of the square world (from 0 to world_size).
            num_obstacles: The number of static obstacles.
            robot_radius: The radius of the point robot for collision checking.
            obstacle_radius: The radius of the obstacles. If float, all obstacles have this radius.
                             If list/array, must match num_obstacles.
            goal_radius: The radius of the goal area for determining success.
            max_velocity: The maximum absolute velocity allowed in x and y directions.
            dt: The time step duration for simulation.
            max_episode_steps: Maximum steps before truncation (informational, use wrapper).
            num_envs: Number of parallel environments.
            num_actions: Dimension of the action space.
            device: PyTorch device ('cpu' or 'cuda').
        """
        super().__init__()
        self.noise_level = noise_level
        self.world_size = float(world_size)
        self.num_obstacles = int(num_obstacles)
        self.robot_radius = float(robot_radius)
        self.goal_radius = float(goal_radius)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.dt = float(dt)
        self._max_episode_steps = max_episode_steps
        self.num_agents = int(num_agents)
        self._num_parallel_envs = int(num_envs)
        self.num_envs = self._num_parallel_envs * self.num_agents
        self.device = device
        self.num_actions = num_actions
        self.max_episode_length = (
            max_episode_steps if max_episode_steps is not None else 1000
        )
        self.use_cbf_action_filtering = use_cbf_action_filtering
        self.use_cbf_reward_penalty = use_cbf_reward_penalty
        if dynamics_model not in ("dynamic", "quasi_static"):
            raise ValueError(f"dynamics_model must be 'dynamic' or 'quasi_static', got '{dynamics_model}'")
        self.dynamics_model = dynamics_model
        if obs_layout not in self.OBS_LAYOUTS:
            raise ValueError(
                f"obs_layout must be one of {self.OBS_LAYOUTS}, got '{obs_layout}'"
            )
        self.obs_layout = obs_layout
        if control_noise not in self.CONTROL_NOISE_LEVELS:
            raise ValueError(
                f"control_noise must be one of {self.CONTROL_NOISE_LEVELS}, got '{control_noise}'"
            )
        self.control_noise = control_noise
        self.control_noise_std = self.CONTROL_NOISE_STD[control_noise]

        # Handle obstacle radii - Store as tensor (num_envs, num_obstacles)
        if isinstance(obstacle_radius, (list, np.ndarray, torch.Tensor)):
            # Expecting shape (num_envs, num_obstacles) or (num_obstacles,) to broadcast
            obstacle_radii_np = (
                np.array(obstacle_radius, dtype=np.float32)
                if not isinstance(obstacle_radius, torch.Tensor)
                else obstacle_radius.cpu().numpy()
            )
            if obstacle_radii_np.ndim == 1:  # Shape (num_obstacles,)
                if len(obstacle_radii_np) != self.num_obstacles:
                    raise ValueError(
                        f"Length of 1D obstacle_radius ({len(obstacle_radii_np)}) "
                        f"must match num_obstacles ({self.num_obstacles})"
                    )
                # Broadcast to (num_envs, num_obstacles)
                self._obstacle_radii = (
                    torch.from_numpy(obstacle_radii_np)
                    .float()
                    .unsqueeze(0)
                    .expand(self._num_parallel_envs, -1)
                    .to(self.device)
                )
            elif obstacle_radii_np.ndim == 2:  # Shape (num_parallel_envs, num_obstacles)
                if obstacle_radii_np.shape != (self._num_parallel_envs, self.num_obstacles):
                    raise ValueError(
                        f"Shape of 2D obstacle_radius {obstacle_radii_np.shape} "
                        f"must match (num_parallel_envs, num_obstacles) ({self._num_parallel_envs}, {self.num_obstacles})"
                    )
                self._obstacle_radii = (
                    torch.from_numpy(obstacle_radii_np).float().to(self.device)
                )
            else:
                raise ValueError(
                    "obstacle_radius must be a float, 1D array/list/tensor, or 2D array/list/tensor"
                )

        else:  # Float case
            self._obstacle_radii = torch.full(
                (self._num_parallel_envs, self.num_obstacles),
                float(obstacle_radius),
                dtype=torch.float32,
                device=self.device,
            )

        # Max radius across all envs and obstacles
        self._max_obstacle_radius = (
            torch.max(self._obstacle_radii).item() if self.num_obstacles > 0 else 0.0
        )

        # --- Remove Gymnasium spaces ---
        # self.action_space = ...
        # self.observation_space = ...

        # --- Internal state variables (vectorized tensors) ---
        # P = _num_parallel_envs, A = num_agents
        self._robot_pos: Optional[torch.Tensor] = None  # shape: (P, A, 2)
        self._goal_pos: Optional[torch.Tensor] = None  # shape: (P, A, 2)
        self._obstacle_positions: Optional[torch.Tensor] = (
            None  # shape: (P, num_obstacles, 2)
        )
        self._robot_vel: torch.Tensor = torch.zeros(
            (self._num_parallel_envs, self.num_agents, 2), dtype=torch.float32, device=self.device
        )
        self._elapsed_steps: Optional[torch.Tensor] = None  # shape: (num_envs,) = (P*A,)
        self.episode_length_buf: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )

        # --- Rendering setup ---
        self.render_mode = render_mode
        self.figure: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.robot_patch: Optional[patches.Circle] = None
        self.goal_patch: Optional[patches.Circle] = None
        self.obstacle_patches: List[patches.Circle] = None
        # Store obstacle radii as numpy for rendering patch creation and reset logic
        self._obstacle_radii_np = (
            self._obstacle_radii.cpu().numpy()
        )  # Shape (num_envs, num_obstacles)
        self._episode_reward_components = {
            'reward_goal': torch.zeros(self.num_envs, device=self.device),
            'reward_obstacle_collision': torch.zeros(self.num_envs, device=self.device),
            'reward_wall_collision': torch.zeros(self.num_envs, device=self.device),
            'reward_progress': torch.zeros(self.num_envs, device=self.device),
            'reward_step_penalty': torch.zeros(self.num_envs, device=self.device),
        }

        # Tracks whether each agent has reached its goal this world-episode.
        # Success fires only when every agent in a world has reached its goal.
        self._agents_reached_goal_buf: torch.Tensor = torch.zeros(
            (self._num_parallel_envs, self.num_agents), dtype=torch.bool, device=self.device
        )
        # Tracks which agents have individually terminated (goal/collision/timeout)
        # but whose world episode has not ended yet.  Frozen agents stay at their
        # current position with zero velocity until the world resets.
        self._world_agents_done_buf: torch.Tensor = torch.zeros(
            (self._num_parallel_envs, self.num_agents), dtype=torch.bool, device=self.device
        )

        self.reset()  # Call reset to initialize state tensors

    # @property
    # def num_steps_per_env(self):
    #     return self.max_episode_length

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:  # Return tensor obs
        """
        Resets the environment to an initial state.

        Args:
            seed: The random seed for reproducibility (for NumPy placement).
            options: Additional options (e.g., specifying start/goal positions).

        Returns:
            A tuple containing the initial observation tensor and extras dictionary.
        """
        if seed is not None:
            np.random.seed(seed)  # Seed NumPy for placement
            # Consider seeding torch as well if torch.rand is used elsewhere
            # torch.manual_seed(seed)
            self.np_random = np.random
        else:
            self.np_random = np.random

        P, A = self._num_parallel_envs, self.num_agents

        # Reset state tensors
        self._agents_reached_goal_buf = torch.zeros(
            (P, A), dtype=torch.bool, device=self.device
        )
        self._world_agents_done_buf = torch.zeros(
            (P, A), dtype=torch.bool, device=self.device
        )
        self._robot_vel = torch.zeros(
            (P, A, 2), dtype=torch.float32, device=self.device
        )
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self.episode_length_buf = torch.zeros_like(
            self._elapsed_steps, dtype=torch.int32
        )

        # Initialize state tensors
        self._robot_pos = torch.zeros(
            (P, A, 2), dtype=torch.float32, device=self.device
        )
        self._goal_pos = torch.zeros(
            (P, A, 2), dtype=torch.float32, device=self.device
        )
        self._obstacle_positions = torch.zeros(
            (P, self.num_obstacles, 2),
            dtype=torch.float32,
            device=self.device,
        )

        # Use NumPy for placement logic, then convert to tensors
        robot_pos_np = np.zeros((P, A, 2), dtype=np.float32)
        goal_pos_np = np.zeros((P, A, 2), dtype=np.float32)
        obstacle_positions_np = np.zeros(
            (P, self.num_obstacles, 2), dtype=np.float32
        )
        obstacle_radii_np = (
            self._obstacle_radii_np
        )  # (P, num_obstacles)

        for env_idx in range(P):
            placement_attempts = 0
            max_placement_attempts = 100  # Increased attempts might be needed
            valid_placement = False
            min_obstacle_separation_buffer = (
                0.5  # Define minimum buffer between obstacles
            )
            min_robot_goal_distance = (
                self.world_size / 3.0
            )  # Minimum distance between robot and goal start
            # Get radii for the current environment
            current_env_obstacle_radii = obstacle_radii_np[
                env_idx
            ]  # Shape (num_obstacles,)
            current_env_max_obstacle_radius = (
                np.max(current_env_obstacle_radii) if self.num_obstacles > 0 else 0.0
            )

            while not valid_placement and placement_attempts < max_placement_attempts:
                placement_attempts += 1

                # 1. Place Obstacles
                if options and options.get("obstacle_pos") is not None:
                    obs_pos = np.array(options["obstacle_pos"], dtype=np.float32)
                elif self.num_obstacles > 0:
                    obs_pos = self.np_random.uniform(
                        current_env_max_obstacle_radius,
                        self.world_size - current_env_max_obstacle_radius,
                        size=(self.num_obstacles, 2),
                    ).astype(np.float32)
                else:
                    obs_pos = np.empty((0, 2), dtype=np.float32)

                # 2. Check Obstacles vs Obstacles
                obstacles_clear = True
                if self.num_obstacles > 1:
                    for i in range(self.num_obstacles):
                        for j in range(i + 1, self.num_obstacles):
                            dist_sq = np.sum((obs_pos[i] - obs_pos[j]) ** 2)
                            min_dist_sq = (
                                current_env_obstacle_radii[i]
                                + current_env_obstacle_radii[j]
                                + min_obstacle_separation_buffer
                            ) ** 2
                            if dist_sq < min_dist_sq:
                                obstacles_clear = False
                                break
                        if not obstacles_clear:
                            break
                if not obstacles_clear:
                    continue

                # 3. Place each agent's goal and robot sequentially
                agent_robots = []
                agent_goals = []
                all_agents_placed = True

                for agent_idx in range(A):
                    goal_wall_buffer = max(self.goal_radius, self.robot_radius)
                    robot_wall_buffer = self.robot_radius + 1.0 + 1e-4
                    wall_buffer = self.robot_radius + 1.0

                    # --- Place goal for this agent ---
                    if options and options.get("goal_pos") is not None:
                        goal_pos = np.array(options["goal_pos"], dtype=np.float32)
                    else:
                        goal_pos = self.np_random.uniform(
                            goal_wall_buffer,
                            self.world_size - goal_wall_buffer,
                            size=(2,),
                        ).astype(np.float32)

                    # Validate goal
                    goal_hsafe = True
                    if (
                        (goal_pos[0] < self.robot_radius)
                        or (goal_pos[0] > self.world_size - self.robot_radius)
                        or (goal_pos[1] < self.robot_radius)
                        or (goal_pos[1] > self.world_size - self.robot_radius)
                    ):
                        goal_hsafe = False
                    if goal_hsafe and self.num_obstacles > 0:
                        for i, o_pos in enumerate(obs_pos):
                            if np.linalg.norm(goal_pos - o_pos) < (
                                self.robot_radius + current_env_obstacle_radii[i]
                            ):
                                goal_hsafe = False
                                break
                    if goal_hsafe and self.num_obstacles > 0:
                        for i, o_pos in enumerate(obs_pos):
                            if np.linalg.norm(goal_pos - o_pos) < (
                                self.goal_radius + current_env_obstacle_radii[i]
                            ):
                                goal_hsafe = False
                                break
                    if not goal_hsafe:
                        all_agents_placed = False
                        break

                    # --- Place robot for this agent ---
                    if options and options.get("robot_pos") is not None:
                        robot_pos = np.array(options["robot_pos"], dtype=np.float32)
                    else:
                        robot_pos = self.np_random.uniform(
                            robot_wall_buffer,
                            self.world_size - robot_wall_buffer,
                            size=(2,),
                        ).astype(np.float32)

                    # Wall clearance
                    if (
                        (robot_pos[0] <= wall_buffer)
                        or (robot_pos[0] >= self.world_size - wall_buffer)
                        or (robot_pos[1] <= wall_buffer)
                        or (robot_pos[1] >= self.world_size - wall_buffer)
                    ):
                        all_agents_placed = False
                        break

                    # Clearance from static obstacles
                    robot_clear = True
                    for i, o_pos in enumerate(obs_pos):
                        if (
                            np.linalg.norm(robot_pos - o_pos)
                            <= self.robot_radius + current_env_obstacle_radii[i] + 1.0
                        ):
                            robot_clear = False
                            break
                    if not robot_clear:
                        all_agents_placed = False
                        break

                    # Clearance from already-placed agents
                    for prev_robot in agent_robots:
                        if np.linalg.norm(robot_pos - prev_robot) <= 2 * self.robot_radius + 1.0:
                            robot_clear = False
                            break
                    if not robot_clear:
                        all_agents_placed = False
                        break

                    # Minimum robot-to-goal distance
                    if np.linalg.norm(robot_pos - goal_pos) < min_robot_goal_distance:
                        all_agents_placed = False
                        break

                    # Require obstacle on path for agent 0 only (ensures non-trivial task)
                    if agent_idx == 0 and self.num_obstacles > 0:
                        dist_robot_goal = np.linalg.norm(robot_pos - goal_pos)
                        obstacle_blocks_path = False
                        for i, o_pos in enumerate(obs_pos):
                            if (
                                abs(
                                    np.linalg.norm(robot_pos - o_pos)
                                    + np.linalg.norm(goal_pos - o_pos)
                                    - dist_robot_goal
                                )
                                < current_env_obstacle_radii[i] * 2
                            ):
                                obstacle_blocks_path = True
                                break
                        if not obstacle_blocks_path:
                            all_agents_placed = False
                            break

                    agent_robots.append(robot_pos)
                    agent_goals.append(goal_pos)

                if all_agents_placed:
                    valid_placement = True

            if not valid_placement:
                print(
                    f"Warning: Failed to find valid initial placement for env {env_idx} after {max_placement_attempts} attempts."
                )
                # Fall back: place agents at default positions if retry exhausted
                if len(agent_robots) < A:
                    for k in range(len(agent_robots), A):
                        agent_robots.append(np.array([self.world_size * 0.2, self.world_size * (0.2 + k * 0.1)], dtype=np.float32))
                        agent_goals.append(np.array([self.world_size * 0.8, self.world_size * (0.8 - k * 0.1)], dtype=np.float32))

            robot_pos_np[env_idx] = np.array(agent_robots, dtype=np.float32)   # (A, 2)
            goal_pos_np[env_idx] = np.array(agent_goals, dtype=np.float32)     # (A, 2)
            if self.num_obstacles > 0:
                obstacle_positions_np[env_idx] = obs_pos

        # Convert placed positions to tensors and assign to state variables
        self._robot_pos = torch.from_numpy(robot_pos_np).to(self.device)
        self._goal_pos = torch.from_numpy(goal_pos_np).to(self.device)
        self._obstacle_positions = torch.from_numpy(obstacle_positions_np).to(
            self.device
        )

        # Get observations and info (now using tensor methods)
        obs, extras = self.get_observations()

        if self.render_mode == "human":
            self._render_frame()

        return obs, extras  # Return tensor obs and dict extras

    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """Constructs the observation dictionary with tensors.

        All tensors are flat (num_envs = P*A, ...) where virtual env vi = e*A+a
        corresponds to agent a in parallel world e.
        """
        if (
            self._robot_pos is None
            or self._goal_pos is None
            or self._robot_vel is None
            or self._obstacle_positions is None
        ):
            raise RuntimeError(
                "Environment state is uninitialized. Call reset() before using the environment."
            )

        P, A = self._num_parallel_envs, self.num_agents
        N = P * A  # total virtual envs

        # Flatten agent dim into batch dim: (P, A, 2) -> (P*A, 2)
        robot_pos_flat = self._robot_pos.reshape(N, 2)
        goal_pos_flat = self._goal_pos.reshape(N, 2)
        robot_vel_flat = self._robot_vel.reshape(N, 2)

        vel_key = self._OBS_VEL_KEY[self.obs_layout]
        obs_dict = {
            "robot_pos": robot_pos_flat,
            "goal_pos": goal_pos_flat,
            vel_key: robot_vel_flat,
        }

        # Static obstacles: repeat obstacle data for each agent in the same world
        # _obstacle_positions: (P, O, 2) -> expand to (P, A, O, 2) -> reshape (P*A, O, 2)
        if self.num_obstacles > 0:
            obs_pos_flat = (
                self._obstacle_positions
                .unsqueeze(1)
                .expand(P, A, self.num_obstacles, 2)
                .reshape(N, self.num_obstacles, 2)
            )
            obs_radii_flat = (
                self._obstacle_radii
                .unsqueeze(1)
                .expand(P, A, self.num_obstacles)
                .reshape(N, self.num_obstacles)
            )
            # Find closest obstacle per virtual env
            rel_pos = obs_pos_flat - robot_pos_flat.unsqueeze(1)
            distances = torch.linalg.norm(rel_pos, dim=2)
            idx_min = torch.argmin(distances, dim=1)
            arange_N = torch.arange(N, device=self.device)
            closest_pos = obs_pos_flat[arange_N, idx_min]        # (N, 2)
            closest_rad = obs_radii_flat[arange_N, idx_min]      # (N,)
            obstacle_info = torch.cat(
                [closest_pos, closest_rad.unsqueeze(1)], dim=1
            ).unsqueeze(1)                                         # (N, 1, 3)
            obs_dict["obstacles"] = obstacle_info.view(N, -1)     # (N, 3)
        else:
            obs_dict["obstacles"] = torch.empty(
                (N, 0), dtype=torch.float32, device=self.device
            )

        # Other agents: each agent sees the other A-1 agents as dynamic obstacles
        if A > 1:
            # _robot_pos: (P, A, 2) -> for agent a, others are all j != a
            others_list = []
            for a in range(A):
                others = torch.cat(
                    [self._robot_pos[:, :a, :], self._robot_pos[:, a+1:, :]], dim=1
                )  # (P, A-1, 2)
                others_list.append(others)
            # Stack over agent dimension: (P, A, A-1, 2) -> reshape (P*A, A-1, 2)
            others_pos = torch.stack(others_list, dim=1).reshape(N, A - 1, 2)
            others_radii = torch.full(
                (N, A - 1, 1), self.robot_radius, dtype=torch.float32, device=self.device
            )
            obs_dict["other_agents"] = torch.cat(
                [others_pos, others_radii], dim=2
            ).reshape(N, (A - 1) * 3)                             # (N, (A-1)*3)

        return obs_dict

    def _get_info(self) -> Dict[str, Any]:
        """Provides auxiliary information about the environment state (tensors where applicable)."""
        if (
            self._robot_pos is None
            or self._goal_pos is None
            or self._obstacle_positions is None
            or self._elapsed_steps is None
        ):
            raise RuntimeError(
                "Environment state is uninitialized. Call reset() before using the environment."
            )

        P, A = self._num_parallel_envs, self.num_agents
        N = P * A
        robot_pos_flat = self._robot_pos.reshape(N, 2)
        goal_pos_flat = self._goal_pos.reshape(N, 2)
        dist_to_goal = torch.linalg.norm(robot_pos_flat - goal_pos_flat, dim=1)

        # Repeat obstacle positions for each agent in the same world
        obs_pos_flat = (
            self._obstacle_positions
            .unsqueeze(1)
            .expand(P, A, self.num_obstacles, 2)
            .reshape(N, self.num_obstacles, 2)
        )

        return {
            "distance_to_goal": dist_to_goal,
            "robot_position": robot_pos_flat.clone(),
            "goal_position": goal_pos_flat.clone(),
            "obstacle_positions": obs_pos_flat.clone(),
            "elapsed_steps": self._elapsed_steps.clone(),
        }

    def get_observations(self):
        """
        Returns the current observations and extras in the format expected by rsl_rl VecEnv:
        - obs: torch.Tensor of shape (num_envs, obs_dim) on self.device
        - extras: dict with key 'observations' containing the observation dict (tensors on self.device)
                  and other info (tensors on self.device)
        """
        obs_dict = self._get_obs()  # Returns dict of tensors
        # Flatten and concatenate all observation components for each env
        obs_list = []
        # Ensure consistent order for concatenation
        for k in sorted(obs_dict.keys()):
            # Ensure tensors are 2D (num_envs, feature_dim) before concatenating
            tensor = obs_dict[k]
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            elif tensor.ndim > 2:  # Should not happen with current obs space
                tensor = tensor.view(self.num_envs, -1)
            obs_list.append(tensor)

        obs = torch.cat(obs_list, dim=1)  # shape (num_envs, obs_dim)

        # Get info dict (already contains tensors where applicable)
        info_dict = self._get_info()

        extras = {"observations": obs_dict}  # obs_dict already contains tensors
        extras.update(info_dict)  # info_dict already contains tensors
        return obs, extras

    def h_function(
        self,
        robot_pos: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the combined CBF h-function using signed distance to obstacles and walls.
        h is the minimum across all active constraints (obstacles and 4 walls).
        Positive => safe, 0 => on boundary, negative => unsafe.
        """
        device = robot_pos.device
        num_envs = robot_pos.shape[0]

        # Obstacles part (handle num_obstacles == 0 safely)
        if obstacle_pos is not None and obstacle_pos.shape[1] > 0:
            # distances: (num_envs, num_obstacles)
            distances = torch.linalg.norm(robot_pos.unsqueeze(1) - obstacle_pos, dim=2)
            # h for each obstacle: d - (r_robot + r_obstacle)
            h_obs_all = distances - (self.robot_radius + obstacle_radius)
            # min across obstacles
            h_obs, _ = torch.min(h_obs_all, dim=1)  # (num_envs,)
        else:
            # No obstacles -> do not constrain via obstacles
            h_obs = torch.full((num_envs,), float("inf"), device=device, dtype=robot_pos.dtype)

        # Walls part (distance from robot center to walls minus robot radius)
        x = robot_pos[:, 0]
        y = robot_pos[:, 1]
        h_left = x - self.robot_radius
        h_right = (self.world_size - x) - self.robot_radius
        h_bottom = y - self.robot_radius
        h_top = (self.world_size - y) - self.robot_radius
        h_wall = torch.minimum(torch.minimum(h_left, h_right), torch.minimum(h_bottom, h_top))

        # Combine: active constraint is the most critical one (minimum h)
        h = torch.minimum(h_obs, h_wall)  # (num_envs,)
        return h

    def gradient_h_function(
        self, robot_pos: torch.Tensor, obstacle_pos: torch.Tensor,
        obstacle_radius: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the gradient of the active CBF constraint (obstacle or wall).
        - For obstacles: unit vector from closest obstacle center to the robot.
        - For walls: outward unit normal of the closest wall.
        """
        device = robot_pos.device
        dtype = robot_pos.dtype
        num_envs = robot_pos.shape[0]

        if obstacle_radius is None:
            obstacle_radius = self._obstacle_radii

        # Compute obstacle-side h and gradient candidates (if any obstacles)
        has_obstacles = obstacle_pos is not None and obstacle_pos.shape[1] > 0
        if has_obstacles:
            # distances: (num_envs, num_obstacles)
            distances = torch.linalg.norm(robot_pos.unsqueeze(1) - obstacle_pos, dim=2)
            h_obs_all = distances - (self.robot_radius + obstacle_radius)
            # pick the obstacle with minimum h
            min_h_obs, min_idx = torch.min(h_obs_all, dim=1)  # (num_envs,), (num_envs,)
            closest_obs = obstacle_pos[torch.arange(num_envs, device=device), min_idx]  # (num_envs,2)
            denom = distances[torch.arange(num_envs, device=device), min_idx].clamp_min(1e-8).unsqueeze(1)
            grad_obs = (robot_pos - closest_obs) / denom  # (num_envs,2)
        else:
            # No obstacles: set min_h_obs to +inf and grad_obs dummy
            min_h_obs = torch.full((num_envs,), float("inf"), device=device, dtype=dtype)
            grad_obs = torch.zeros((num_envs, 2), device=device, dtype=dtype)

        # Compute wall h values and gradients
        x = robot_pos[:, 0]
        y = robot_pos[:, 1]
        h_vals = torch.stack(
            [
                x - self.robot_radius,                           # left wall (grad [1,0])
                (self.world_size - x) - self.robot_radius,       # right wall (grad [-1,0])
                y - self.robot_radius,                           # bottom wall (grad [0,1])
                (self.world_size - y) - self.robot_radius,       # top wall (grad [0,-1])
            ],
            dim=1,
        )  # (num_envs, 4)
        min_h_wall, wall_idx = torch.min(h_vals, dim=1)  # (num_envs,), (num_envs,)

        wall_grads_lut = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            device=device,
            dtype=dtype,
        )  # order matches h_vals stacking
        grad_wall = wall_grads_lut[wall_idx]  # (num_envs,2)

        # Select active gradient based on which constraint is more critical
        use_obstacle = min_h_obs <= min_h_wall
        grad_h = torch.where(use_obstacle.unsqueeze(1), grad_obs, grad_wall)
        return grad_h

    def _hessian_vel_term(
        self,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes velᵀ ∇²h vel for the active CBF constraint.
        Obstacles: (||vel||² - (∇h·vel)²) / dist  (curvature of signed distance)
        Walls: 0  (linear constraints, zero Hessian)
        """
        device = robot_pos.device
        dtype = robot_pos.dtype
        num_envs = robot_pos.shape[0]

        if obstacle_radius is None:
            obstacle_radius = self._obstacle_radii

        has_obstacles = obstacle_pos is not None and obstacle_pos.shape[1] > 0
        if has_obstacles:
            distances = torch.linalg.norm(robot_pos.unsqueeze(1) - obstacle_pos, dim=2)
            h_obs_all = distances - (self.robot_radius + obstacle_radius)
            min_h_obs, min_idx = torch.min(h_obs_all, dim=1)
            d_active = distances[torch.arange(num_envs, device=device), min_idx].clamp_min(1e-8)
            closest_obs = obstacle_pos[torch.arange(num_envs, device=device), min_idx]
            n_obs = (robot_pos - closest_obs) / d_active.unsqueeze(1)
            v_dot_n = torch.sum(robot_vel * n_obs, dim=1)
            lf2h_obs = (torch.sum(robot_vel ** 2, dim=1) - v_dot_n ** 2) / d_active
        else:
            min_h_obs = torch.full((num_envs,), float("inf"), device=device, dtype=dtype)
            lf2h_obs = torch.zeros(num_envs, device=device, dtype=dtype)

        x, y = robot_pos[:, 0], robot_pos[:, 1]
        h_walls = torch.stack([
            x - self.robot_radius,
            (self.world_size - x) - self.robot_radius,
            y - self.robot_radius,
            (self.world_size - y) - self.robot_radius,
        ], dim=1)
        min_h_wall, _ = torch.min(h_walls, dim=1)

        if has_obstacles:
            use_obs = min_h_obs <= min_h_wall
            return torch.where(use_obs, lf2h_obs, torch.zeros(num_envs, device=device, dtype=dtype))
        return torch.zeros(num_envs, device=device, dtype=dtype)

    def filter_velocity(
        self,
        robot_pos: torch.Tensor,
        vel: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters velocity for single-integrator (quasi-static) dynamics using a relative-degree-1 CBF.

        Enforces: ∇h · vel + alpha * h ≥ 0

        Returns (filtered_vel, psi) where psi = ∇h · vel + alpha * h before filtering.
        psi > 0 means safe.
        """
        h = self.h_function(robot_pos, obstacle_pos, obstacle_radius)  # (num_envs,)
        grad_h = self.gradient_h_function(robot_pos, obstacle_pos, obstacle_radius)  # (num_envs, 2)

        Lfh = torch.sum(grad_h * vel, dim=1)   # ∇h · vel
        psi = Lfh + alpha * h                  # (num_envs,)

        filtered_vel = vel.clone()
        violated = torch.where(psi < 0)[0]
        if violated.numel() > 0:
            denom = torch.sum(grad_h[violated] ** 2, dim=1).clamp_min(1e-12)
            correction = (-psi[violated] / denom).unsqueeze(1) * grad_h[violated]
            filtered_vel[violated] += correction

        return filtered_vel, psi

    def filter_acceleration(
        self,
        robot_pos: torch.Tensor,
        robot_vel: torch.Tensor,
        accel: torch.Tensor,
        obstacle_pos: torch.Tensor,
        obstacle_radius: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters acceleration for double-integrator dynamics using a relative-degree-2 CBF.

        Defines ψ₁ = ∇h·vel + h.
        Enforces ψ̇₁ + ψ₁ ≥ 0, which expands to:
            ∇h·acc + velᵀ∇²h·vel + 2(∇h·vel) + h ≥ 0

        Returns (filtered_accel, ψ₁). ψ₁ > 0 means safe.
        """
        h = self.h_function(robot_pos, obstacle_pos, obstacle_radius)                          # (num_envs,)
        grad_h = self.gradient_h_function(robot_pos, obstacle_pos, obstacle_radius)          # (num_envs, 2)
        lf2h = self._hessian_vel_term(robot_pos, robot_vel, obstacle_pos, obstacle_radius)  # (num_envs,)

        Lfh = torch.sum(grad_h * robot_vel, dim=1)                            # ∇h·vel
        psi1 = Lfh + h                                                        # composite barrier

        # slack < 0 means the desired accel violates the constraint
        slack = torch.sum(grad_h * accel, dim=1) + lf2h + 2 * Lfh + h       # (num_envs,)

        filtered_accel = accel.clone()
        filtered_ids = torch.where(slack < 0)[0]
        if filtered_ids.numel() > 0:
            denom = torch.sum(grad_h[filtered_ids] ** 2, dim=1).clamp_min(1e-12)
            correction = (-slack[filtered_ids] / denom).unsqueeze(1) * grad_h[filtered_ids]
            filtered_accel[filtered_ids] += correction

        return filtered_accel, psi1

    def step(self, action: torch.Tensor):
        """
        Executes one time step in the environment using PyTorch tensors.

        action: (num_envs, num_actions) = (P*A, num_actions)
        Returns obs (P*A, obs_dim), reward (P*A,), done (P*A,), extras.
        """
        if action.device != self.device:
            action = action.to(self.device)

        P, A = self._num_parallel_envs, self.num_agents
        N = P * A

        # prev distances before update: (P, A)
        prev_dist_to_goal = torch.linalg.norm(self._robot_pos - self._goal_pos, dim=2)

        # Reshape action (P*A, 2) -> (P, A, 2) for per-agent processing
        action_3d = action.reshape(P, A, self.num_actions)

        new_pos_list = []
        new_vel_list = []
        psi_list = []
        filtered_list = []
        clipped_list = []

        for a in range(A):
            agent_pos = self._robot_pos[:, a, :]   # (P, 2)
            agent_vel = self._robot_vel[:, a, :]   # (P, 2)
            agent_act = action_3d[:, a, :]         # (P, 2)

            # Frozen agents (individually done but world episode not over) stay put.
            is_frozen = self._world_agents_done_buf[:, a]  # (P,)
            agent_act = torch.where(is_frozen.unsqueeze(1), torch.zeros_like(agent_act), agent_act)

            # Extend obstacle set with other agents as dynamic obstacles
            if A > 1:
                other_pos = torch.cat(
                    [self._robot_pos[:, :a, :], self._robot_pos[:, a+1:, :]], dim=1
                )  # (P, A-1, 2)
                other_radii = torch.full(
                    (P, A - 1), self.robot_radius, dtype=torch.float32, device=self.device
                )
                ext_obs_pos = torch.cat([self._obstacle_positions, other_pos], dim=1)
                ext_obs_radii = torch.cat([self._obstacle_radii, other_radii], dim=1)
            else:
                ext_obs_pos = self._obstacle_positions
                ext_obs_radii = self._obstacle_radii

            if self.dynamics_model == "quasi_static":
                clipped = torch.clamp(agent_act, -self.max_velocity, self.max_velocity)
                filtered, psi = self.filter_velocity(
                    agent_pos, clipped, ext_obs_pos, ext_obs_radii
                )
                applied = filtered if self.use_cbf_action_filtering else clipped
                # Actuation noise on the velocity command (post-filter): the safe
                # control is corrupted before it physically moves the robot.
                if self.control_noise_std > 0.0:
                    applied = applied + torch.randn_like(applied) * self.max_velocity * self.control_noise_std
                noise = torch.randn_like(applied) * self.max_velocity * self.noise_level
                new_pos = torch.clamp(
                    agent_pos + (applied + noise) * self.dt, 0.0, self.world_size
                )
                new_vel = applied
            else:
                clipped = torch.clamp(agent_act, -self.max_acceleration, self.max_acceleration)
                filtered, psi = self.filter_acceleration(
                    agent_pos, agent_vel, clipped, ext_obs_pos, ext_obs_radii
                )
                applied = filtered if self.use_cbf_action_filtering else clipped
                # Actuation noise on the acceleration command (post-filter).
                if self.control_noise_std > 0.0:
                    applied = applied + torch.randn_like(applied) * self.max_acceleration * self.control_noise_std
                new_vel = torch.clamp(
                    agent_vel + applied * self.dt, -self.max_velocity, self.max_velocity
                )
                noise = torch.randn_like(new_vel) * self.max_velocity * self.noise_level
                new_pos = torch.clamp(
                    agent_pos + (new_vel + noise) * self.dt, 0.0, self.world_size
                )

            # Frozen agents: restore original position and zero velocity
            new_pos = torch.where(is_frozen.unsqueeze(1), agent_pos, new_pos)
            new_vel = torch.where(is_frozen.unsqueeze(1), torch.zeros_like(new_vel), new_vel)
            clipped = torch.where(is_frozen.unsqueeze(1), torch.zeros_like(clipped), clipped)
            filtered = torch.where(is_frozen.unsqueeze(1), torch.zeros_like(filtered), filtered)
            psi = torch.where(is_frozen, torch.zeros_like(psi), psi)

            new_pos_list.append(new_pos)
            new_vel_list.append(new_vel)
            psi_list.append(psi)
            filtered_list.append(filtered)
            clipped_list.append(clipped)

        # Update state: stack per-agent results -> (P, A, 2)
        self._robot_pos = torch.stack(new_pos_list, dim=1)
        self._robot_vel = torch.stack(new_vel_list, dim=1)
        # Only increment elapsed steps for non-frozen agents
        not_frozen_flat = ~self._world_agents_done_buf.reshape(N)
        self._elapsed_steps[not_frozen_flat] += 1
        self.episode_length_buf = self._elapsed_steps.clone().int()

        # Flat views for termination / reward (all in N = P*A space)
        robot_pos_flat = self._robot_pos.reshape(N, 2)
        goal_pos_flat = self._goal_pos.reshape(N, 2)
        prev_dist_flat = prev_dist_to_goal.reshape(N)
        current_dist_flat = torch.linalg.norm(robot_pos_flat - goal_pos_flat, dim=1)

        # psi, filtered, clipped: stack (P,) per agent -> (P, A) -> (N,)
        psi_flat = torch.stack(psi_list, dim=1).reshape(N) if not isinstance(psi_list[0], int) else torch.zeros(N, device=self.device)
        filtered_flat = torch.stack(filtered_list, dim=1).reshape(N, self.num_actions)
        clipped_flat = torch.stack(clipped_list, dim=1).reshape(N, self.num_actions)

        # --- Termination ---
        terminated = torch.zeros(N, dtype=torch.bool, device=self.device)
        collided_obstacle = torch.zeros(N, dtype=torch.bool, device=self.device)
        wall_collision = torch.zeros(N, dtype=torch.bool, device=self.device)

        # Static obstacle collision (flat)
        if self.num_obstacles > 0:
            obs_pos_flat = (
                self._obstacle_positions
                .unsqueeze(1).expand(P, A, self.num_obstacles, 2)
                .reshape(N, self.num_obstacles, 2)
            )
            obs_radii_flat = (
                self._obstacle_radii
                .unsqueeze(1).expand(P, A, self.num_obstacles)
                .reshape(N, self.num_obstacles)
            )
            dist_sq = torch.sum(
                (robot_pos_flat.unsqueeze(1) - obs_pos_flat) ** 2, dim=2
            )
            thresh_sq = (self.robot_radius + obs_radii_flat) ** 2
            collided_obstacle = torch.any(dist_sq < thresh_sq, dim=1)
            terminated |= collided_obstacle

        # Agent-agent collision (skip pairs where either agent is already frozen at its goal)
        if A > 1:
            arange_P = torch.arange(P, device=self.device)
            for ai in range(A):
                for aj in range(ai + 1, A):
                    dist_ij = torch.linalg.norm(
                        self._robot_pos[:, ai, :] - self._robot_pos[:, aj, :], dim=1
                    )  # (P,)
                    collision_ij = dist_ij < 2 * self.robot_radius
                    # Frozen agents have already completed their task — ignore collisions with them.
                    collision_ij = collision_ij & ~self._world_agents_done_buf[:, ai] & ~self._world_agents_done_buf[:, aj]
                    flat_i = arange_P * A + ai
                    flat_j = arange_P * A + aj
                    terminated[flat_i] |= collision_ij
                    terminated[flat_j] |= collision_ij
                    collided_obstacle[flat_i] |= collision_ij
                    collided_obstacle[flat_j] |= collision_ij

        # Wall collision
        wall_collision = (
            (robot_pos_flat[:, 0] < self.robot_radius)
            | (robot_pos_flat[:, 0] > self.world_size - self.robot_radius)
            | (robot_pos_flat[:, 1] < self.robot_radius)
            | (robot_pos_flat[:, 1] > self.world_size - self.robot_radius)
        )
        wall_collision = wall_collision & (~terminated)
        terminated |= wall_collision

        # Goal check
        goal_reached = current_dist_flat < (self.robot_radius + self.goal_radius)
        goal_reached = goal_reached & (~terminated)
        terminated |= goal_reached

        # --- Rewards ---
        active_mask = ~terminated
        reward_goal = goal_reached.float()
        reward_obstacle_collision = collided_obstacle.float() * -1.0
        reward_wall_collision = wall_collision.float() * -1.0
        max_progress_per_step = self.max_velocity * self.dt + 1e-8
        reward_progress = (
            20 * (prev_dist_flat - current_dist_flat) / max_progress_per_step
            * active_mask.float()
        )
        reward_step_penalty = -0.01 * active_mask.float()

        if self.use_cbf_reward_penalty:
            reward_clipped_action = 100 * (
                torch.exp(
                    -torch.sum(torch.square(clipped_flat - filtered_flat), dim=1) / 0.25
                ) * active_mask.float() - 1
            )
            reward_psi = torch.clamp(psi_flat, max=0.0) * active_mask.float() * 10
        else:
            reward_clipped_action = torch.zeros(N, device=self.device)
            reward_psi = torch.zeros(N, device=self.device)

        reward = (
            reward_goal
            + reward_obstacle_collision
            + reward_wall_collision
            + reward_progress
            + reward_step_penalty
            + reward_clipped_action
            + reward_psi
        )
        reward_log = reward - reward_psi

        # --- Truncation (only for non-frozen agents) ---
        truncated = torch.zeros(N, dtype=torch.bool, device=self.device)
        if self._max_episode_steps is not None:
            truncated = self._elapsed_steps >= self._max_episode_steps
        # Don't re-truncate already-frozen agents
        truncated = truncated & not_frozen_flat
        timeout_mask = truncated & (~terminated)
        reward_timeout = timeout_mask.float() * -10.0
        reward = reward + reward_timeout

        # --- World-level termination ---
        # Mark agents that just individually terminated this step.
        agent_individually_done = terminated | truncated
        self._world_agents_done_buf |= agent_individually_done.reshape(P, A)

        # Update goal-reach buffer (needed for success check).
        self._agents_reached_goal_buf |= goal_reached.reshape(P, A)

        # A world fails when any agent has a non-goal termination.
        world_any_fail = (
            (collided_obstacle | wall_collision | timeout_mask).reshape(P, A)
        ).any(dim=1)  # (P,)

        # World episode ends when every agent has individually terminated OR any agent failed.
        world_episode_done = self._world_agents_done_buf.all(dim=1) | world_any_fail  # (P,)

        # World success = world done AND all agents reached goal AND no failures.
        world_success = world_episode_done & self._agents_reached_goal_buf.all(dim=1) & ~world_any_fail

        # Clear buffers for worlds whose episode just ended.
        self._world_agents_done_buf[world_episode_done] = False
        self._agents_reached_goal_buf[world_episode_done | world_any_fail] = False

        # Expand to flat (P*A,) for logging and done signal.
        world_success_flat = world_success.unsqueeze(1).expand(P, A).reshape(N)
        world_episode_done_flat = world_episode_done.unsqueeze(1).expand(P, A).reshape(N)

        # done fires for ALL agents in a world simultaneously.
        done = world_episode_done_flat
        self.episode_length_buf[done] = 0
        self._elapsed_steps[done] = 0

        # --- Prepare outputs ---
        obs, extras = self.get_observations()

        extras["log"] = {
            "reward_goal": reward_goal,
            "reward_obstacle_collision": reward_obstacle_collision,
            "reward_wall_collision": reward_wall_collision,
            "reward_progress": reward_progress,
            "reward_step_penalty": reward_step_penalty,
            "reward_psi": reward_psi,
            "reward_clipped_action": reward_clipped_action,
            "reward_timeout": reward_timeout,
            "success": world_success_flat.float(),
            "collided_obstacle": collided_obstacle.float(),
            "collided_wall": wall_collision.float(),
            "psi": psi_flat,
        }
        extras["episode"] = {
            "reward": reward,
            "reward_log": reward_log,
            "length": self.episode_length_buf,
            "reward_goal": reward_goal,
            "reward_obstacle_collision": reward_obstacle_collision,
            "reward_wall_collision": reward_wall_collision,
            "reward_progress": reward_progress,
            "reward_step_penalty": reward_step_penalty,
            "reward_clipped_action": reward_clipped_action,
            "reward_psi": reward_psi,
            "reward_timeout": reward_timeout,
            "success": world_success_flat.float(),
            "collided_obstacle": collided_obstacle.float(),
            "collided_wall": wall_collision.float(),
        }

        reset_indices = torch.where(done)[0]
        if len(reset_indices) > 0:
            self._reset_envs(reset_indices)

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, done, extras

    # --- Rendering methods ---
    def render(self) -> Optional[np.ndarray]:
        """
        Renders the environment. Converts tensors to NumPy for plotting.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self) -> Optional[np.ndarray]:
        """Internal rendering function using Matplotlib."""
        if self.render_mode is None:
            print(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization."
            )
            return None

        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("matplotlib is not installed, run `pip install matplotlib`")
            return None

        if self.figure is None:  # Initialize plot on first call
            if self.render_mode == "human":
                plt.ion()  # Enable interactive mode for human rendering
                self.figure, self.ax = plt.subplots(figsize=(6, 6))
            elif self.render_mode == "rgb_array":
                # No interactive mode needed for array rendering
                self.figure, self.ax = plt.subplots(figsize=(6, 6))

            self.ax.set_xlim(0, self.world_size)
            self.ax.set_ylim(0, self.world_size)
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_title("Custom Navigation Environment")
            self.ax.set_xlabel("X Position")
            self.ax.set_ylabel("Y Position")

            # Create patches (visual elements) only once
            # One robot + goal patch per agent; colors cycle through blue shades
            agent_colors = ["blue", "royalblue", "deepskyblue", "steelblue",
                            "dodgerblue", "cornflowerblue", "mediumblue", "navy"]
            goal_colors = ["green", "limegreen", "forestgreen", "darkgreen",
                           "mediumseagreen", "springgreen", "olivedrab", "teal"]
            A = self.num_agents
            self.robot_patch = [
                patches.Circle(
                    (0, 0), self.robot_radius,
                    fc=agent_colors[a % len(agent_colors)], alpha=0.8,
                    label=f"Robot {a}" if a == 0 else f"Robot {a}",
                )
                for a in range(A)
            ]
            self.goal_patch = [
                patches.Circle(
                    (0, 0), self.goal_radius,
                    fc=goal_colors[a % len(goal_colors)], alpha=0.8,
                    label=f"Goal {a}" if a == 0 else f"Goal {a}",
                )
                for a in range(A)
            ]
            first_env_radii_np = self._obstacle_radii_np[0]
            self.obstacle_patches = [
                patches.Circle((0, 0), radius, fc="red", alpha=0.6)
                for radius in first_env_radii_np
            ]

            for p in self.robot_patch:
                self.ax.add_patch(p)
            for p in self.goal_patch:
                self.ax.add_patch(p)
            for patch in self.obstacle_patches:
                self.ax.add_patch(patch)
            self.ax.legend(loc="upper right")

        # --- Update patch positions based on current state ---
        if (
            self._robot_pos is None
            or self._goal_pos is None
            or self._obstacle_positions is None
        ):
            print("Warning: Attempting to render before reset() or with invalid state.")
            if self.render_mode == "rgb_array":
                return np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                return None

        # Render world 0: _robot_pos[0] is (A, 2), _goal_pos[0] is (A, 2)
        robot_pos_np = self._robot_pos[0].cpu().numpy()       # (A, 2)
        goal_pos_np = self._goal_pos[0].cpu().numpy()          # (A, 2)
        obstacle_positions_np = self._obstacle_positions[0].cpu().numpy()  # (O, 2)

        for a in range(self.num_agents):
            self.robot_patch[a].set_center(robot_pos_np[a])
            self.goal_patch[a].set_center(goal_pos_np[a])
        for i, patch in enumerate(self.obstacle_patches):
            if i < self.num_obstacles:
                patch.set_center(obstacle_positions_np[i])
            else:
                patch.set_center((0, 0))

        # --- Draw and return/display (unchanged) ---
        if self.render_mode == "human":
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            # plt.pause(0.001) # Small pause helps update plot in some backends
            return None
        elif self.render_mode == "rgb_array":
            self.figure.canvas.draw()
            image = np.frombuffer(self.figure.canvas.tostring_rgb(), dtype="uint8")
            width, height = self.figure.canvas.get_width_height()
            image = image.reshape(height, width, 3)
            # Note: Closing the figure here would prevent updates in subsequent calls.
            # Keep the figure object alive unless explicitly closing the env.
            return image
        else:
            return None  # Should not happen if render_mode is validated
    
    def _reset_world_layout(
        self,
        world_e: int,
        robot_pos_np: np.ndarray,
        goal_pos_np: np.ndarray,
        obstacle_positions_np: np.ndarray,
        obstacle_radii_np: np.ndarray,
        options: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Sample a fresh obstacle + agent layout for one parallel world.

        Matches the initial ``reset()`` placement logic. Used for episode-end
        resets so single-agent training behaves like performance-test (full
        world re-randomization) and multi-agent worlds reset all agents together
        with new shared obstacles.
        """
        A = self.num_agents
        placement_attempts = 0
        max_placement_attempts = 100
        valid_placement = False
        min_obstacle_separation_buffer = 0.5
        min_robot_goal_distance = self.world_size / 3.0
        current_env_obstacle_radii = obstacle_radii_np[world_e]
        current_env_max_obstacle_radius = (
            np.max(current_env_obstacle_radii) if self.num_obstacles > 0 else 0.0
        )
        agent_robots: List[np.ndarray] = []
        agent_goals: List[np.ndarray] = []
        obs_pos = np.empty((0, 2), dtype=np.float32)

        while not valid_placement and placement_attempts < max_placement_attempts:
            placement_attempts += 1
            agent_robots = []
            agent_goals = []

            if options and options.get("obstacle_pos") is not None:
                obs_pos = np.array(options["obstacle_pos"], dtype=np.float32)
            elif self.num_obstacles > 0:
                obs_pos = self.np_random.uniform(
                    current_env_max_obstacle_radius,
                    self.world_size - current_env_max_obstacle_radius,
                    size=(self.num_obstacles, 2),
                ).astype(np.float32)
            else:
                obs_pos = np.empty((0, 2), dtype=np.float32)

            obstacles_clear = True
            if self.num_obstacles > 1:
                for i in range(self.num_obstacles):
                    for j in range(i + 1, self.num_obstacles):
                        dist_sq = np.sum((obs_pos[i] - obs_pos[j]) ** 2)
                        min_dist_sq = (
                            current_env_obstacle_radii[i]
                            + current_env_obstacle_radii[j]
                            + min_obstacle_separation_buffer
                        ) ** 2
                        if dist_sq < min_dist_sq:
                            obstacles_clear = False
                            break
                    if not obstacles_clear:
                        break
            if not obstacles_clear:
                continue

            all_agents_placed = True
            for agent_idx in range(A):
                goal_wall_buffer = max(self.goal_radius, self.robot_radius)
                robot_wall_buffer = self.robot_radius + 1.0 + 1e-4
                wall_buffer = self.robot_radius + 1.0

                if options and options.get("goal_pos") is not None:
                    goal_pos = np.array(options["goal_pos"], dtype=np.float32)
                else:
                    goal_pos = self.np_random.uniform(
                        goal_wall_buffer,
                        self.world_size - goal_wall_buffer,
                        size=(2,),
                    ).astype(np.float32)

                goal_hsafe = True
                if (
                    (goal_pos[0] < self.robot_radius)
                    or (goal_pos[0] > self.world_size - self.robot_radius)
                    or (goal_pos[1] < self.robot_radius)
                    or (goal_pos[1] > self.world_size - self.robot_radius)
                ):
                    goal_hsafe = False
                if goal_hsafe and self.num_obstacles > 0:
                    for i, o_pos in enumerate(obs_pos):
                        if np.linalg.norm(goal_pos - o_pos) < (
                            self.robot_radius + current_env_obstacle_radii[i]
                        ):
                            goal_hsafe = False
                            break
                if goal_hsafe and self.num_obstacles > 0:
                    for i, o_pos in enumerate(obs_pos):
                        if np.linalg.norm(goal_pos - o_pos) < (
                            self.goal_radius + current_env_obstacle_radii[i]
                        ):
                            goal_hsafe = False
                            break
                if not goal_hsafe:
                    all_agents_placed = False
                    break

                if options and options.get("robot_pos") is not None:
                    robot_pos = np.array(options["robot_pos"], dtype=np.float32)
                else:
                    robot_pos = self.np_random.uniform(
                        robot_wall_buffer,
                        self.world_size - robot_wall_buffer,
                        size=(2,),
                    ).astype(np.float32)

                if (
                    (robot_pos[0] <= wall_buffer)
                    or (robot_pos[0] >= self.world_size - wall_buffer)
                    or (robot_pos[1] <= wall_buffer)
                    or (robot_pos[1] >= self.world_size - wall_buffer)
                ):
                    all_agents_placed = False
                    break

                robot_clear = True
                for i, o_pos in enumerate(obs_pos):
                    if (
                        np.linalg.norm(robot_pos - o_pos)
                        <= self.robot_radius + current_env_obstacle_radii[i] + 1.0
                    ):
                        robot_clear = False
                        break
                if not robot_clear:
                    all_agents_placed = False
                    break

                for prev_robot in agent_robots:
                    if np.linalg.norm(robot_pos - prev_robot) <= 2 * self.robot_radius + 1.0:
                        robot_clear = False
                        break
                if not robot_clear:
                    all_agents_placed = False
                    break

                if np.linalg.norm(robot_pos - goal_pos) < min_robot_goal_distance:
                    all_agents_placed = False
                    break

                if agent_idx == 0 and self.num_obstacles > 0:
                    dist_robot_goal = np.linalg.norm(robot_pos - goal_pos)
                    obstacle_blocks_path = False
                    for i, o_pos in enumerate(obs_pos):
                        if (
                            abs(
                                np.linalg.norm(robot_pos - o_pos)
                                + np.linalg.norm(goal_pos - o_pos)
                                - dist_robot_goal
                            )
                            < current_env_obstacle_radii[i] * 2
                        ):
                            obstacle_blocks_path = True
                            break
                    if not obstacle_blocks_path:
                        all_agents_placed = False
                        break

                agent_robots.append(robot_pos)
                agent_goals.append(goal_pos)

            if all_agents_placed:
                valid_placement = True

        if not valid_placement:
            return False

        if len(agent_robots) < A:
            for k in range(len(agent_robots), A):
                agent_robots.append(
                    np.array(
                        [self.world_size * 0.2, self.world_size * (0.2 + k * 0.1)],
                        dtype=np.float32,
                    )
                )
                agent_goals.append(
                    np.array(
                        [self.world_size * 0.8, self.world_size * (0.8 - k * 0.1)],
                        dtype=np.float32,
                    )
                )

        robot_pos_np[world_e] = np.array(agent_robots, dtype=np.float32)
        goal_pos_np[world_e] = np.array(agent_goals, dtype=np.float32)
        if self.num_obstacles > 0:
            obstacle_positions_np[world_e] = obs_pos
        return True

    def _reset_envs(self, reset_indices: torch.Tensor):
        """
        Resets parallel worlds whose episodes ended.

        reset_indices are virtual env ids in [0, P*A). Worlds are deduplicated so
        each finished episode triggers one full-world layout refresh (obstacles +
        all agents), matching performance-test for num_agents=1 and giving
        comparable multi-agent training resets.
        """
        if not isinstance(reset_indices, torch.Tensor):
            reset_indices = torch.tensor(reset_indices, dtype=torch.long)
        reset_indices_np = reset_indices.cpu().numpy()

        A = self.num_agents
        world_indices = np.unique(reset_indices_np // A)

        robot_pos_np = self._robot_pos.cpu().numpy()
        goal_pos_np = self._goal_pos.cpu().numpy()
        obstacle_positions_np = self._obstacle_positions.cpu().numpy()
        obstacle_radii_np = self._obstacle_radii.cpu().numpy()

        for world_e in world_indices:
            world_e = int(world_e)
            if not self._reset_world_layout(
                world_e,
                robot_pos_np,
                goal_pos_np,
                obstacle_positions_np,
                obstacle_radii_np,
            ):
                print(
                    f"Warning: Failed to reset world {world_e} after 100 attempts."
                )

        device = self.device
        self._robot_pos = torch.from_numpy(robot_pos_np).to(device)
        self._goal_pos = torch.from_numpy(goal_pos_np).to(device)
        self._obstacle_positions = torch.from_numpy(obstacle_positions_np).to(device)

        self._robot_vel[world_indices] = 0
        self._world_agents_done_buf[world_indices] = False
        self._agents_reached_goal_buf[world_indices] = False
        flat_indices = np.concatenate(
            [np.arange(w * A, (w + 1) * A, dtype=np.int64) for w in world_indices]
        )
        self._elapsed_steps[flat_indices] = 0
        self.episode_length_buf[flat_indices] = 0

    def close(self):
        """Cleans up resources, like the rendering window."""
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
            self.robot_patch = None
            self.goal_patch = None
            self.obstacle_patches = None
            if plt.isinteractive():
                plt.ioff()


# add main function for testing
if __name__ == "__main__":
    env = UnifiedNavigationEnv(
        render_mode="human", device="cpu", num_envs=1
    )  # Specify device, test with 1 env for render
    obs, extras = env.reset()  # Use the VecEnv API return format (obs is tensor)
    # done is now a tensor
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for _ in range(env.max_episode_length * 2):  # Run for longer to see resets
        if done.any():  # Check if any env is done
            print("Resetting environment(s)")
            # Note: In a real scenario, the runner handles resets based on 'done'.
            # Here, we manually break or could call reset again if needed.
            # For simplicity, just break the loop or let it run.
            # obs, extras = env.reset() # Optional: reset manually if not handled externally
            # done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            pass  # Let the loop continue to show post-reset behavior if internal reset happens

        # Generate random actions as torch tensor on the correct device
        # Use torch.rand for tensor-native random generation
        action = (
            torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1
        ) * env.max_velocity
        obs, reward, done, extras = env.step(action)  # Use VecEnv API (all tensors)

        # Rendering still uses numpy arrays from internal state (first env)
        env.render()

        # Optional: Print step info
        # print(f"Step: {extras['elapsed_steps'][0].item()}, Reward: {reward[0].item():.2f}, Done: {done[0].item()}")

        if done.all():  # Exit if all envs are done
            print("All environments finished.")
            break

    env.close()
    for _ in range(env.max_episode_length * 2):  # Run for longer to see resets
        if done.any():  # Check if any env is done
            print("Resetting environment(s)")
            # Note: In a real scenario, the runner handles resets based on 'done'.
            # Here, we manually break or could call reset again if needed.
            # For simplicity, just break the loop or let it run.
            # obs, extras = env.reset() # Optional: reset manually if not handled externally
            # done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            pass  # Let the loop continue to show post-reset behavior if internal reset happens

        # Generate random actions as torch tensor on the correct device
        # Use torch.rand for tensor-native random generation
        action = (
            torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1
        ) * env.max_velocity
        obs, reward, done, extras = env.step(action)  # Use VecEnv API (all tensors)

        # Rendering still uses numpy arrays from internal state (first env)
        env.render()

        # Optional: Print step info
        # print(f"Step: {extras['elapsed_steps'][0].item()}, Reward: {reward[0].item():.2f}, Done: {done[0].item()}")

        if done.all():  # Exit if all envs are done
            print("All environments finished.")
            break

    env.close()
