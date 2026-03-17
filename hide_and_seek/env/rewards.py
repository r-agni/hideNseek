"""Reward functions for hide-and-seek RL training.

Each function has the signature:
    def reward_fn(env: ManagerBasedRLEnv) -> torch.Tensor
and returns a tensor of shape (num_envs,).

Detection is proximity-based: seeker must physically reach the hider (≤0.5 m).

Weights applied in HideAndSeekEnv._compute_step_rewards() and RewardsCfg:
    seeker_detection        10.0   sparse: seeker reaches hider (≤0.5 m confirmed)
    seeker_approach          2.0   dense: proximity_signal guides seeker toward hider
    seeker_alive             0.01  per-step: stay upright
    hider_survival           1.0   per-step: remain undetected during seeking
    hider_distance           0.5   dense: reward hider for staying far from seeker
    hider_hiding_movement    0.2   dense: move during hiding phase (find cover)
    hider_alive              0.01  per-step: stay upright
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Seeker rewards
# ---------------------------------------------------------------------------

def seeker_detection_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse +1 reward when seeker camera confirms detection of the hider."""
    return env.hider_detected.float()


def seeker_approach_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense reward guiding the seeker toward the hider.

    proximity_signal = 1 - dist / (2 × DETECTION_RADIUS), clamped to [0, 1].
    Peaks at 1.0 when seeker is on top of hider, fades to 0 at 2× detection range.
    Only active during SEEKING.
    """
    seeking = env.phase_manager.is_seeking()
    return torch.where(seeking, env.proximity_signal, torch.zeros_like(env.proximity_signal))


def seeker_alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-step bonus for seeker staying upright (pelvis above 0.65 m).

    G1 stands at ~0.93 m. 0.65 m threshold catches fallen/collapsing robots
    while still being above the termination floor (0.3 m).
    """
    upright = env.scene["seeker"].data.root_pos_w[:, 2] > 0.65
    return upright.float()


# ---------------------------------------------------------------------------
# Hider rewards
# ---------------------------------------------------------------------------

def hider_survival_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-step reward for hider remaining undetected during seeking phase."""
    seeking      = env.phase_manager.is_seeking()
    not_detected = ~env.hider_detected
    return (seeking & not_detected).float()


def hider_distance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense reward for hider staying far from the seeker.

    Reward = (1 - proximity_signal) during seeking — opposite of seeker's approach
    signal. Maximum when seeker is far; zero when seeker is on top.
    Only active during SEEKING phase.
    """
    seeking = env.phase_manager.is_seeking()
    far     = 1.0 - env.proximity_signal
    return torch.where(seeking, far, torch.zeros_like(far))


def hider_hiding_movement_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense reward for moving during the hiding phase.

    Reward = min(|v_body|, 1.0) — linear velocity magnitude in body frame.
    Breaks the stand-still-in-place local minimum: hider must navigate to
    cover during the 5-second hiding window before seeking begins.
    Zero during seeking (survival + occlusion signals take over).
    """
    hiding = env.phase_manager.is_hiding()
    speed  = env.scene["hider"].data.root_lin_vel_b.norm(dim=-1).clamp(0.0, 1.0)
    return torch.where(hiding, speed, torch.zeros_like(speed))


def hider_alive_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-step bonus for hider staying upright (pelvis above 0.65 m)."""
    upright = env.scene["hider"].data.root_pos_w[:, 2] > 0.65
    return upright.float()
