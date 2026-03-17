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

# G1 robot pelvis height at normal standing is ~0.93 m.
# Below 0.3 m means the robot has fallen and collapsed on the ground.
_FALLEN_HEIGHT_THRESHOLD = 0.3  # meters


def hider_detected(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode ends when seeker confirms detection of hider."""
    return env.hider_detected


def game_phase_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode ends when seeking phase times out (DONE phase reached)."""
    return env.phase_manager.is_done()


def robot_fallen(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode ends if either robot's pelvis drops below the fallen threshold.

    This catches cases where a robot tips over, falls, or is otherwise in an
    unrecoverable pose that would make further simulation meaningless.
    """
    seeker_fallen = env.scene["seeker"].data.root_pos_w[:, 2] < _FALLEN_HEIGHT_THRESHOLD
    hider_fallen  = env.scene["hider"].data.root_pos_w[:, 2] < _FALLEN_HEIGHT_THRESHOLD
    return seeker_fallen | hider_fallen
