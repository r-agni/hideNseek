"""Termination conditions for hide-and-seek episodes.

Each function has the signature:
    def term_fn(env: ManagerBasedRLEnv) -> torch.Tensor
and returns a boolean tensor of shape (num_envs,).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hider_detected(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode ends when seeker confirms detection of hider."""
    return env.hider_detected


def game_phase_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode ends when game reaches DONE phase (seeking timeout)."""
    return env.phase_manager.is_done()


def out_of_bounds(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode ends if either agent falls below the scene floor."""
    seeker = env.scene["seeker"]
    hider = env.scene["hider"]
    floor_threshold = -1.0  # meters below origin
    seeker_oob = seeker.data.root_pos_w[:, 2] < floor_threshold
    hider_oob = hider.data.root_pos_w[:, 2] < floor_threshold
    return seeker_oob | hider_oob
