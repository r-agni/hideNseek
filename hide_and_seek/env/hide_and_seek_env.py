"""Hide-and-seek environment with two Unitree G1 robots.

Extends Isaac Lab's ManagerBasedRLEnv with game phase logic,
visibility detection, and per-phase action masking.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from hide_and_seek.env.actions import apply_base_velocity
from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg
from hide_and_seek.game.phase_manager import PhaseManager
from hide_and_seek.game.visibility import (
    VisibilityTracker,
    check_visibility_frustum,
)


class HideAndSeekEnv(ManagerBasedRLEnv):
    """Two-agent hide-and-seek environment built on Isaac Lab.

    Game flow:
        1. INIT: Both agents are spawned at random positions.
        2. HIDING: Hider can move; seeker is frozen.
        3. SEEKING: Both agents can move; seeker tries to spot hider.
        4. DONE: Episode ends (hider found or timeout).

    The environment manages two G1 robots ("seeker" and "hider") in an
    indoor scene, with head-mounted cameras for egocentric observations.
    """

    cfg: HideAndSeekEnvCfg

    def __init__(self, cfg: HideAndSeekEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Game state managers
        self.phase_manager = PhaseManager(
            num_envs=self.num_envs,
            hiding_steps=cfg.hiding_phase_steps,
            seeking_steps=cfg.seeking_phase_steps,
            device=self.device,
        )
        self.visibility_tracker = VisibilityTracker(
            num_envs=self.num_envs,
            confirmation_steps=cfg.detection_confirmation_steps,
            device=self.device,
        )

        # Detection state
        self.hider_detected = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions with phase-based masking.

        During HIDING phase, seeker actions are zeroed out.
        During DONE phase, all actions are zeroed out.

        Args:
            actions: Raw action tensor of shape (num_envs, action_dim).
                First 2 dims = seeker [fwd_vel, turn_vel],
                next 2 dims = hider [fwd_vel, turn_vel].
        """
        # Split actions for each agent
        seeker_actions = actions[:, 0:2].clone()
        hider_actions = actions[:, 2:4].clone()

        # Phase-based masking
        hiding = self.phase_manager.is_hiding()
        done = self.phase_manager.is_done()

        # Freeze seeker during hiding phase
        seeker_actions[hiding] = 0.0
        # Freeze both during done phase
        seeker_actions[done] = 0.0
        hider_actions[done] = 0.0

        # Apply velocity commands
        apply_base_velocity(self, seeker_actions, "seeker")
        apply_base_velocity(self, hider_actions, "hider")

    def _post_physics_step(self):
        """Advance game phase and check visibility after physics."""
        # Step phase manager
        self.phase_manager.step()

        # Check visibility only during seeking phase
        seeking = self.phase_manager.is_seeking()
        if seeking.any():
            self._check_visibility(seeking)

    def _check_visibility(self, seeking_mask: torch.Tensor):
        """Run visibility detection for environments in seeking phase.

        Uses frustum-based detection. If semantic camera data is available,
        it can be swapped in for more accurate pixel-based detection.
        """
        seeker = self.scene["seeker"]
        hider = self.scene["hider"]

        seeker_pos = seeker.data.root_pos_w
        hider_pos = hider.data.root_pos_w

        # Compute seeker forward direction from quaternion
        seeker_quat = seeker.data.root_quat_w  # (num_envs, 4) as (w, x, y, z)
        seeker_forward = _quat_to_forward(seeker_quat)

        # Frustum check
        is_visible = check_visibility_frustum(
            seeker_pos=seeker_pos,
            seeker_forward=seeker_forward,
            hider_pos=hider_pos,
            fov_deg=90.0,
            max_distance=10.0,
        )

        # Only count visibility during seeking phase
        is_visible = is_visible & seeking_mask

        # Update tracker (requires consecutive frames)
        confirmed = self.visibility_tracker.update(is_visible)
        self.hider_detected = self.hider_detected | confirmed

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        super()._reset_idx(env_ids)
        self.hider_detected[env_ids] = False
        self.phase_manager.reset(env_ids)
        self.visibility_tracker.reset(env_ids)


def _quat_to_forward(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to forward direction vector.

    Assumes forward is along +X axis in the robot's local frame.

    Args:
        quat: Quaternion tensor of shape (N, 4) in (w, x, y, z) format.

    Returns:
        Forward direction vectors of shape (N, 3).
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Forward vector (+X) rotated by quaternion
    forward = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y + w * z),
            2 * (x * z - w * y),
        ],
        dim=-1,
    )
    return forward
