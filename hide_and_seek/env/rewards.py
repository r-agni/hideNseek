"""Reward functions for hide-and-seek (placeholder for training phase).

Each function has the signature:
    def reward_fn(env: ManagerBasedRLEnv) -> torch.Tensor
and returns a tensor of shape (num_envs,).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def seeker_detection_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for seeker when hider is detected."""
    return env.hider_detected.float()


def hider_survival_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for hider for each step it remains undetected during seeking."""
    seeking = env.phase_manager.is_seeking()
    not_detected = ~env.hider_detected
    return (seeking & not_detected).float()


def seeker_approach_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for seeker getting closer to hider during seeking phase."""
    seeker = env.scene["seeker"]
    hider = env.scene["hider"]
    distance = torch.norm(
        hider.data.root_pos_w - seeker.data.root_pos_w, dim=-1
    )
    seeking = env.phase_manager.is_seeking()
    # Negative distance as reward (closer = higher reward), only during seeking
    return torch.where(seeking, -distance / 10.0, torch.zeros_like(distance))
