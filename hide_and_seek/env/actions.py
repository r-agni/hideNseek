"""Action configurations for hide-and-seek agents.

Both agents use velocity-based base commands (linear + angular velocity).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Action dimensions per agent
# [linear_velocity_x, angular_velocity_z]
ACTION_DIM = 2

# Velocity limits
MAX_LINEAR_VEL = 1.5  # m/s
MAX_ANGULAR_VEL = 2.0  # rad/s


def apply_base_velocity(
    env: ManagerBasedRLEnv,
    actions: torch.Tensor,
    agent_name: str,
    max_linear: float = MAX_LINEAR_VEL,
    max_angular: float = MAX_ANGULAR_VEL,
):
    """Apply velocity commands to an agent's base.

    Actions are in [-1, 1] and scaled to velocity limits.

    Args:
        env: Environment instance.
        actions: Tensor of shape (num_envs, 2) — [fwd_vel, turn_vel].
        agent_name: Name of the agent in the scene ("seeker" or "hider").
        max_linear: Maximum linear velocity in m/s.
        max_angular: Maximum angular velocity in rad/s.
    """
    agent = env.scene[agent_name]

    # Scale actions from [-1, 1] to velocity range
    lin_vel = actions[:, 0:1] * max_linear
    ang_vel = actions[:, 1:2] * max_angular

    # Build velocity command: (lin_x, lin_y, lin_z, ang_x, ang_y, ang_z)
    num_envs = actions.shape[0]
    velocity = torch.zeros(num_envs, 6, device=actions.device)
    velocity[:, 0] = lin_vel.squeeze(-1)  # Forward (X)
    velocity[:, 5] = ang_vel.squeeze(-1)  # Yaw (around Z)

    agent.write_root_velocity_to_sim(velocity)
