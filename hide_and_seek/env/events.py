"""Event/randomization functions for hide-and-seek episodes.

Called on reset to randomize agent spawn positions and orientations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_agents_to_random_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    min_spawn_distance: float = 3.0,
    spawn_height: float = 0.8,
    spawn_range: float = 5.0,
    max_attempts: int = 100,
):
    """Randomize seeker and hider positions on reset.

    Samples random positions within a bounding box and ensures they
    are at least min_spawn_distance apart.

    Args:
        env: The environment instance.
        env_ids: Indices of environments being reset.
        min_spawn_distance: Minimum distance between agents in meters.
        spawn_height: Height above ground to spawn (G1 is ~1.27m tall).
        spawn_range: Half-extent of spawn area in X and Y.
        max_attempts: Maximum sampling attempts before accepting any valid pair.
    """
    num_resets = len(env_ids)
    device = env_ids.device

    seeker = env.scene["seeker"]
    hider = env.scene["hider"]

    for _ in range(max_attempts):
        # Random XY positions within spawn range
        seeker_xy = (torch.rand(num_resets, 2, device=device) * 2 - 1) * spawn_range
        hider_xy = (torch.rand(num_resets, 2, device=device) * 2 - 1) * spawn_range

        # Check distance constraint
        dist = torch.norm(seeker_xy - hider_xy, dim=-1)
        valid = dist >= min_spawn_distance

        if valid.all():
            break
    else:
        # Accept whatever we have — distance constraint is best-effort
        pass

    # Build full 3D positions
    seeker_pos = torch.zeros(num_resets, 3, device=device)
    seeker_pos[:, 0] = seeker_xy[:, 0]
    seeker_pos[:, 1] = seeker_xy[:, 1]
    seeker_pos[:, 2] = spawn_height

    hider_pos = torch.zeros(num_resets, 3, device=device)
    hider_pos[:, 0] = hider_xy[:, 0]
    hider_pos[:, 1] = hider_xy[:, 1]
    hider_pos[:, 2] = spawn_height

    # Random orientations (yaw only — rotation around Z axis)
    seeker_yaw = torch.rand(num_resets, device=device) * 2 * 3.14159
    hider_yaw = torch.rand(num_resets, device=device) * 2 * 3.14159

    # Convert yaw to quaternion (w, x, y, z) — rotation around Z
    seeker_quat = _yaw_to_quat(seeker_yaw, device)
    hider_quat = _yaw_to_quat(hider_yaw, device)

    # Apply to articulations
    seeker.write_root_pose_to_sim(
        torch.cat([seeker_pos, seeker_quat], dim=-1), env_ids
    )
    hider.write_root_pose_to_sim(
        torch.cat([hider_pos, hider_quat], dim=-1), env_ids
    )

    # Reset velocities to zero
    zero_vel = torch.zeros(num_resets, 6, device=device)
    seeker.write_root_velocity_to_sim(zero_vel, env_ids)
    hider.write_root_velocity_to_sim(zero_vel, env_ids)

    # Reset game state
    env.phase_manager.reset(env_ids)
    env.visibility_tracker.reset(env_ids)
    env.hider_detected[env_ids] = False


def _yaw_to_quat(yaw: torch.Tensor, device: str) -> torch.Tensor:
    """Convert yaw angles (radians) to quaternions (w, x, y, z).

    Rotation around Z-axis.

    Args:
        yaw: Tensor of shape (N,) with yaw angles.
        device: Torch device.

    Returns:
        Quaternions of shape (N, 4) in (w, x, y, z) format.
    """
    half_yaw = yaw * 0.5
    quat = torch.zeros(len(yaw), 4, device=device)
    quat[:, 0] = torch.cos(half_yaw)  # w
    quat[:, 3] = torch.sin(half_yaw)  # z
    return quat
