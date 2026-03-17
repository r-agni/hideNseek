"""Observation functions for the hide-and-seek environment.

Each function has the signature:
    def obs_fn(env: ManagerBasedRLEnv) -> torch.Tensor
and returns a tensor of shape (num_envs, *obs_dim).

Detection is proximity-based (≤0.5 m). The seeker receives a proximity_signal
∈ [0, 1] that peaks when on top of the hider — no GPS oracle, just a "warmth"
signal that requires the seeker to physically navigate close.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Seeker observations
# ---------------------------------------------------------------------------

def seeker_root_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker root position in world frame, shape (N, 3)."""
    return env.scene["seeker"].data.root_pos_w


def seeker_root_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker root orientation (w, x, y, z), shape (N, 4)."""
    return env.scene["seeker"].data.root_quat_w


def seeker_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker base linear velocity in body frame, shape (N, 3)."""
    return env.scene["seeker"].data.root_lin_vel_b


def seeker_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker base angular velocity in body frame, shape (N, 3)."""
    return env.scene["seeker"].data.root_ang_vel_b


def seeker_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker joint positions, shape (N, num_joints)."""
    return env.scene["seeker"].data.joint_pos


def seeker_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seeker joint velocities, shape (N, num_joints)."""
    return env.scene["seeker"].data.joint_vel


def seeker_detection_signal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Proximity-based detection state, shape (N, 2).

    Index 0: proximity_signal — ∈ [0, 1], peaks at 1 when seeker is on top
             of hider, fades to 0 at 2× detection radius. Zero during HIDING.
             Gives the seeker a "warmth" gradient without GPS coordinates.
    Index 1: hider_confirmed — 0.0 or 1.0, whether detection is confirmed.
    """
    seeking   = env.phase_manager.is_seeking().unsqueeze(-1)   # (N, 1)
    proximity = env.proximity_signal.unsqueeze(-1)             # (N, 1)
    confirmed = env.hider_detected.float().unsqueeze(-1)       # (N, 1)
    proximity_masked = torch.where(seeking, proximity, torch.zeros_like(proximity))
    return torch.cat([proximity_masked, confirmed], dim=-1)    # (N, 2)


# ---------------------------------------------------------------------------
# Hider observations
# ---------------------------------------------------------------------------

def hider_root_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider root position in world frame, shape (N, 3)."""
    return env.scene["hider"].data.root_pos_w


def hider_root_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider root orientation (w, x, y, z), shape (N, 4)."""
    return env.scene["hider"].data.root_quat_w


def hider_lin_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider base linear velocity in body frame, shape (N, 3)."""
    return env.scene["hider"].data.root_lin_vel_b


def hider_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider base angular velocity in body frame, shape (N, 3)."""
    return env.scene["hider"].data.root_ang_vel_b


def hider_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider joint positions, shape (N, num_joints)."""
    return env.scene["hider"].data.joint_pos


def hider_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Hider joint velocities, shape (N, num_joints)."""
    return env.scene["hider"].data.joint_vel


# ---------------------------------------------------------------------------
# Shared game state observations
# ---------------------------------------------------------------------------

def game_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current game phase as float, shape (N, 1). 0=INIT 1=HIDING 2=SEEKING 3=DONE."""
    return env.phase_manager.phase.unsqueeze(-1).float()


def phase_timer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Normalized time remaining in current phase ∈ [0, 1], shape (N, 1)."""
    return env.phase_manager.get_normalized_timer().unsqueeze(-1)
