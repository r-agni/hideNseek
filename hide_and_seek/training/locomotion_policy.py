"""Pretrained G1 locomotion policy wrapper.

Wraps the Unitree G1 TorchScript JIT model from unitree_rl_gym
(deploy/pre_train/g1/motion.pt) to produce joint position targets
from a high-level velocity command.

The policy input is a 47-dimensional observation vector (from g1_env.py):
  [0:3]   base angular velocity (body frame) × 0.25
  [3:6]   projected gravity × 1.0
  [6:9]   velocity command [vx, vy, yaw_rate] × [2.0, 2.0, 0.25]
  [9:21]  joint position offset from default × 1.0
  [21:33] joint velocity × 0.05
  [33:45] previous actions (raw policy output)
  [45]    sin(2π × phase)
  [46]    cos(2π × phase)

Output: 12 joint position offsets. Applied as:
  joint_target = output × 0.25 + default_joint_pos
"""

from __future__ import annotations

from pathlib import Path

import torch


class LocomotionPolicy:
    """Wraps the pretrained Unitree G1 JIT locomotion policy.

    If the policy file is not found, falls back to holding the default
    joint pose so the rollout loop still runs without crashing.
    """

    # G1 12-DOF joint order (matches unitree_rl_gym g1 config):
    #   left:  hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
    #   right: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll
    # Values from G1_CFG in isaaclab_assets (the same defaults used for spawning):
    #   hip_pitch=-0.20, knee=0.42, ankle_pitch=-0.23, all others=0.0
    G1_DEFAULT_JOINT_POS = torch.tensor([
        0.0,  0.0, -0.20,  0.42, -0.23,  0.0,   # left leg
        0.0,  0.0, -0.20,  0.42, -0.23,  0.0,   # right leg
    ], dtype=torch.float32)

    # Observation scaling factors from unitree_rl_gym g1_env.py
    ANG_VEL_SCALE = 0.25
    # Command scale matches commands_scale = [lin_vel_scale, lin_vel_scale, ang_vel_scale]
    CMD_LIN_SCALE = 2.0
    CMD_ANG_SCALE = 0.25
    DOF_POS_SCALE = 1.0
    DOF_VEL_SCALE = 0.05
    ACTION_SCALE  = 0.25

    # G1 leg joint names (in the order expected by unitree_rl_gym motion.pt)
    G1_LEG_JOINT_NAMES = [
        "left_hip_yaw_joint",   "left_hip_roll_joint",  "left_hip_pitch_joint",
        "left_knee_joint",      "left_ankle_pitch_joint","left_ankle_roll_joint",
        "right_hip_yaw_joint",  "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint",     "right_ankle_pitch_joint","right_ankle_roll_joint",
    ]

    NUM_JOINTS = 12
    OBS_DIM    = 47

    def __init__(self, policy_path: str, num_envs: int, device: str):
        self.device    = device
        self.num_envs  = num_envs
        self.default_joint_pos = self.G1_DEFAULT_JOINT_POS.to(device)
        self._prev_actions = torch.zeros(num_envs, self.NUM_JOINTS, device=device)
        # Will be resolved on first step() call from the robot articulation
        self._leg_joint_indices: "torch.Tensor | None" = None
        # Phase tracker for sin/cos phase (gait cycle, period=0.8s at 60Hz sim / 15Hz control)
        self._phase = torch.zeros(num_envs, device=device)

        self._policy = None
        if policy_path and Path(policy_path).exists():
            self._policy = torch.jit.load(policy_path, map_location=device)
            self._policy.eval()
            print(f"[locomotion] loaded policy from {policy_path}")
        else:
            print(f"[locomotion] policy not found at {policy_path!r} — holding default pose")

    @property
    def loaded(self) -> bool:
        return self._policy is not None

    def _resolve_leg_indices(self, robot) -> torch.Tensor:
        """Resolve the leg joint indices from the articulation on first call."""
        joint_names = robot.data.joint_names
        indices = []
        for name in self.G1_LEG_JOINT_NAMES:
            if name in joint_names:
                indices.append(joint_names.index(name))
            else:
                # Try without "_joint" suffix
                alt = name.replace("_joint", "")
                if alt in joint_names:
                    indices.append(joint_names.index(alt))
                else:
                    raise ValueError(
                        f"[locomotion] Could not find joint '{name}' in articulation. "
                        f"Available joints: {joint_names}"
                    )
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def step(self, robot, velocity_command: torch.Tensor) -> torch.Tensor:
        """Compute joint position targets for one physics step.

        Args:
            robot: Isaac Lab Articulation object (seeker or hider).
            velocity_command: (num_envs, 3) tensor [vx, vy, yaw_rate] in m/s, rad/s.

        Returns:
            joint_targets: (num_envs, 12) absolute joint position targets in radians.
        """
        if self._policy is None:
            return self.default_joint_pos.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Resolve leg joint indices lazily on first call
        if self._leg_joint_indices is None:
            self._leg_joint_indices = self._resolve_leg_indices(robot)
            print(f"[locomotion] leg joint indices: {self._leg_joint_indices.tolist()}")

        # Advance gait phase (period = 0.8s; dt ≈ 1/30 Hz control step)
        self._phase = (self._phase + (1.0 / 30.0) / 0.8) % 1.0

        obs = self._build_obs(robot, velocity_command)
        with torch.no_grad():
            raw = self._policy(obs)  # (num_envs, 12)
        self._prev_actions = raw.clone()
        return raw * self.ACTION_SCALE + self.default_joint_pos

    def _build_obs(self, robot, cmd: torch.Tensor) -> torch.Tensor:
        import math
        idx = self._leg_joint_indices
        ang_vel    = robot.data.root_ang_vel_b * self.ANG_VEL_SCALE          # (N, 3)
        gravity    = robot.data.projected_gravity_b                           # (N, 3)
        dof_pos    = (robot.data.joint_pos[:, idx] - self.default_joint_pos) * self.DOF_POS_SCALE  # (N, 12)
        dof_vel    = robot.data.joint_vel[:, idx] * self.DOF_VEL_SCALE       # (N, 12)

        c = cmd.to(self.device)
        cmd_scaled = torch.stack([
            c[:, 0] * self.CMD_LIN_SCALE,
            c[:, 1] * self.CMD_LIN_SCALE,
            c[:, 2] * self.CMD_ANG_SCALE,
        ], dim=-1)  # (N, 3)

        sin_phase  = torch.sin(2 * math.pi * self._phase).unsqueeze(-1)  # (N, 1)
        cos_phase  = torch.cos(2 * math.pi * self._phase).unsqueeze(-1)  # (N, 1)

        return torch.cat(
            [ang_vel, gravity, cmd_scaled, dof_pos, dof_vel, self._prev_actions, sin_phase, cos_phase],
            dim=-1,
        )  # (N, 47)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset previous-action buffer and phase for the given environment indices."""
        self._prev_actions[env_ids] = 0.0
        self._phase[env_ids] = 0.0
