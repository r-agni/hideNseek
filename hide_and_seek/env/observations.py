"""Observation functions for the hide-and-seek environment.

Each function has the signature:
    def obs_fn(env: ManagerBasedRLEnv) -> torch.Tensor
and returns a tensor of shape (num_envs, *obs_dim).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def seeker_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker robot joint positions (normalized)."""
    seeker = env.scene["seeker"]
    return seeker.data.joint_pos


def seeker_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker robot joint velocities."""
    seeker = env.scene["seeker"]
    return seeker.data.joint_vel


def hider_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider robot joint positions (normalized)."""
    hider = env.scene["hider"]
    return hider.data.joint_pos


def hider_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider robot joint velocities."""
    hider = env.scene["hider"]
    return hider.data.joint_vel


def seeker_root_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker root position in world frame, shape (num_envs, 3)."""
    seeker = env.scene["seeker"]
    return seeker.data.root_pos_w


def hider_root_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider root position in world frame, shape (num_envs, 3)."""
    hider = env.scene["hider"]
    return hider.data.root_pos_w


def seeker_root_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker root orientation as quaternion (w, x, y, z), shape (num_envs, 4)."""
    seeker = env.scene["seeker"]
    return seeker.data.root_quat_w


def hider_root_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider root orientation as quaternion (w, x, y, z), shape (num_envs, 4)."""
    hider = env.scene["hider"]
    return hider.data.root_quat_w


def game_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current game phase as integer, shape (num_envs, 1).

    0=INIT, 1=HIDING, 2=SEEKING, 3=DONE
    """
    return env.phase_manager.phase.unsqueeze(-1).float()


def phase_timer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalized time remaining in current phase, shape (num_envs, 1)."""
    return env.phase_manager.get_normalized_timer().unsqueeze(-1)


def relative_hider_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Position of hider relative to seeker, shape (num_envs, 3).

    Only available during SEEKING phase; zeros during HIDING.
    This is in seeker's local frame.
    """
    seeker = env.scene["seeker"]
    hider = env.scene["hider"]
    relative = hider.data.root_pos_w - seeker.data.root_pos_w

    # Mask during hiding phase (seeker shouldn't know hider position)
    hiding_mask = env.phase_manager.is_hiding().unsqueeze(-1)
    relative = torch.where(hiding_mask, torch.zeros_like(relative), relative)

    return relative
