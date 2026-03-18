"""Hide-and-seek environment with two Unitree G1 robots.

Extends Isaac Lab's ManagerBasedRLEnv with:
  - Game phase logic (HIDING → SEEKING → DONE)
  - Proximity-based detection: seeker detects hider when within DETECTION_RADIUS
    metres (physical contact range). No cameras needed — faster training.
  - Two frozen locomotion policies (motion.pt) that convert RL velocity
    commands into joint position targets each physics step
  - Per-agent reward caching (seeker_step_reward, hider_step_reward) so
    the PPO runner can compute independent GAE and loss for each agent

Action space: 6D continuous ∈ [-1, 1]
  [seeker_vx, seeker_vy, seeker_yaw, hider_vx, hider_vy, hider_yaw]
  Scaled by CMD_SCALE in actions.py before passing to locomotion policy.
"""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

from hide_and_seek.env.actions import CMD_SCALE, apply_joint_targets
from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg
from hide_and_seek.game.phase_manager import PhaseManager
from hide_and_seek.game.visibility import VisibilityTracker

# Seeker must be within this distance (metres) to detect the hider.
# ~0.5 m = touching / directly on top — seeker must physically reach the hider.
DETECTION_RADIUS = 0.5


class HideAndSeekEnv(ManagerBasedRLEnv):
    """Two-agent hide-and-seek environment built on Isaac Lab.

    Game flow:
        1. INIT   → transitions immediately to HIDING
        2. HIDING → hider moves, seeker is frozen (velocity command zeroed)
        3. SEEKING → both agents move, seeker tries to spot hider via camera
        4. DONE   → episode over (hider found or timeout)

    The environment owns two LocomotionPolicy instances (one per agent).
    These are frozen (eval mode) and convert velocity commands → joint targets
    every physics step. The RL policy only learns to choose velocity commands.

    Attributes exposed for the PPO runner:
        hider_detected (bool tensor, N): True once seeker is within DETECTION_RADIUS.
        proximity_signal (float tensor, N): 1 - dist/DETECTION_RADIUS when within
            2×DETECTION_RADIUS, else 0. Dense approach signal for the seeker.
        seeker_step_reward (float tensor, N): Per-step seeker reward (all terms).
        hider_step_reward (float tensor, N): Per-step hider reward (all terms).
    """

    cfg: HideAndSeekEnvCfg

    def __init__(self, cfg: HideAndSeekEnvCfg, **kwargs):
        # Pre-initialize game state managers on cuda:0 so observation functions
        # called inside super().__init__() → load_managers() don't crash on
        # missing attributes. They are re-created on the correct device after.
        import torch as _torch
        _device = "cuda:0" if _torch.cuda.is_available() else "cpu"
        self.phase_manager = PhaseManager(
            num_envs=cfg.scene.num_envs,
            hiding_steps=cfg.hiding_phase_steps,
            seeking_steps=cfg.seeking_phase_steps,
            device=_device,
        )
        self.visibility_tracker = VisibilityTracker(
            num_envs=cfg.scene.num_envs,
            confirmation_steps=cfg.detection_confirmation_steps,
            device=_device,
        )
        self.hider_detected     = _torch.zeros(cfg.scene.num_envs, dtype=_torch.bool,    device=_device)
        self.proximity_signal   = _torch.zeros(cfg.scene.num_envs, dtype=_torch.float32, device=_device)
        self.seeker_step_reward = _torch.zeros(cfg.scene.num_envs, dtype=_torch.float32, device=_device)
        self.hider_step_reward  = _torch.zeros(cfg.scene.num_envs, dtype=_torch.float32, device=_device)

        super().__init__(cfg, **kwargs)

        # Spawn dome light (must be done after sim init, not in InteractiveScene)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/DomeLight", light_cfg)

        # Re-create game state managers on the correct device
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
        self.hider_detected     = torch.zeros(self.num_envs, dtype=torch.bool,    device=self.device)
        self.proximity_signal   = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.seeker_step_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.hider_step_reward  = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # Load frozen locomotion policies (one per agent)
        from hide_and_seek.training.locomotion_policy import LocomotionPolicy
        policy_path = cfg.locomotion_policy_path
        self._seeker_loco = LocomotionPolicy(policy_path, self.num_envs, str(self.device))
        self._hider_loco  = LocomotionPolicy(policy_path, self.num_envs, str(self.device))

        # Debug: print actuator config so we can verify PD gains
        for name in ("seeker", "hider"):
            robot = self.scene[name]
            print(f"[debug] {name} joint_names: {robot.data.joint_names}")
            print(f"[debug] {name} init joint_pos (first env): {robot.data.joint_pos[0].tolist()}")


    # ------------------------------------------------------------------
    # Physics callbacks
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Decode 6D velocity commands → locomotion policy → joint targets."""
        scale = CMD_SCALE.to(self.device)
        seeker_cmd = actions[:, 0:3] * scale
        hider_cmd  = actions[:, 3:6] * scale

        hiding = self.phase_manager.is_hiding()
        done   = self.phase_manager.is_done()
        seeker_cmd[hiding | done] = 0.0
        hider_cmd[done]           = 0.0

        seeker_joints = self._seeker_loco.step(self.scene["seeker"], seeker_cmd)
        hider_joints  = self._hider_loco.step( self.scene["hider"],  hider_cmd)

        s_ids = self._seeker_loco._leg_joint_indices
        h_ids = self._hider_loco._leg_joint_indices
        apply_joint_targets(
            self, seeker_joints, "seeker",
            joint_ids=s_ids.tolist() if s_ids is not None else None,
        )
        apply_joint_targets(
            self, hider_joints, "hider",
            joint_ids=h_ids.tolist() if h_ids is not None else None,
        )

    def _post_physics_step(self):
        """Advance game phase, check visibility, and cache per-agent rewards."""
        self.phase_manager.step()

        seeking = self.phase_manager.is_seeking()
        if seeking.any():
            self._check_visibility(seeking)

        # Cache per-agent rewards so the PPO runner can read them after env.step()
        self._compute_step_rewards()

    def _check_visibility(self, seeking_mask: torch.Tensor):
        """Proximity-based detection: seeker must be within DETECTION_RADIUS metres.

        proximity_signal is a dense approach signal in [0, 1]:
            1.0  when seeker is at the hider's position
            0.0  when seeker is >= 2×DETECTION_RADIUS away
        This gives the seeker a gradient to approach before the sparse detection.
        """
        seeker_pos = self.scene["seeker"].data.root_pos_w[:, :2]  # (N, 2) XY only
        hider_pos  = self.scene["hider"].data.root_pos_w[:, :2]   # (N, 2)

        dist = torch.norm(seeker_pos - hider_pos, dim=-1)          # (N,)

        # Dense approach signal: peaks at 1 when touching, fades to 0 at 2×radius
        approach = (1.0 - dist / (2.0 * DETECTION_RADIUS)).clamp(0.0, 1.0)
        self.proximity_signal = torch.where(
            seeking_mask, approach, torch.zeros_like(approach)
        )

        # Detection: within DETECTION_RADIUS, confirmed after consecutive frames
        is_close  = (dist <= DETECTION_RADIUS) & seeking_mask
        confirmed = self.visibility_tracker.update(is_close)
        self.hider_detected = self.hider_detected | confirmed

    def _compute_step_rewards(self):
        """Compute and cache per-agent rewards for the current step.

        Called every step (after visibility update) so the PPO runner can
        read env.seeker_step_reward and env.hider_step_reward after env.step().
        Rewards are computed here rather than in the runner to keep weights
        in one place (hide_and_seek_env_cfg.py → RewardsCfg).
        """
        from hide_and_seek.env import rewards as _rew
        self.seeker_step_reward = (
            _rew.seeker_detection_reward(self) * 10.0
          + _rew.seeker_approach_reward(self)  *  2.0
          + _rew.seeker_alive_bonus(self)       *  0.01
        )
        self.hider_step_reward = (
            _rew.hider_survival_reward(self)        * 1.0
          + _rew.hider_distance_reward(self)        * 0.5
          + _rew.hider_hiding_movement_reward(self) * 0.2
          + _rew.hider_alive_bonus(self)            * 0.01
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments including locomotion policy state."""
        super()._reset_idx(env_ids)
        self.hider_detected[env_ids]   = False
        self.proximity_signal[env_ids] = 0.0
        self.seeker_step_reward[env_ids]   = 0.0
        self.hider_step_reward[env_ids]    = 0.0
        self.phase_manager.reset(env_ids)
        self.visibility_tracker.reset(env_ids)
        self._seeker_loco.reset(env_ids)
        self._hider_loco.reset(env_ids)


