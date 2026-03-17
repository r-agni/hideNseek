"""Action configurations for hide-and-seek agents.

The RL policy outputs 6D velocity commands (3 per agent):
    [seeker_vx, seeker_vy, seeker_yaw, hider_vx, hider_vy, hider_yaw]
All values are in [-1, 1] and scaled by CMD_SCALE before passing to the
frozen locomotion policy (motion.pt) which converts them to joint targets.

PassthroughAction / PassthroughActionCfg satisfy Isaac Lab's ActionManager
(which requires at least one registered action term) but do no work themselves —
the actual joint application happens in HideAndSeekEnv._pre_physics_step().
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# RL action space constants
# ---------------------------------------------------------------------------

CMD_DIM    = 3   # velocity command dimensions per agent (vx, vy, yaw_rate)
ACTION_DIM = 6   # total action dimensions (CMD_DIM × 2 agents)

# Physical velocity limits — RL action ∈ [-1, 1] is multiplied by these
VX_MAX  = 1.0   # m/s  (forward/backward)
VY_MAX  = 0.5   # m/s  (strafe)
YAW_MAX = 1.0   # rad/s (rotation)

CMD_SCALE = torch.tensor([VX_MAX, VY_MAX, YAW_MAX], dtype=torch.float32)


# ---------------------------------------------------------------------------
# PassthroughAction — no-op placeholder for Isaac Lab's ActionManager
# ---------------------------------------------------------------------------

class PassthroughAction(ActionTerm):
    """No-op action term — actions are decoded and applied in _pre_physics_step."""

    cfg: "PassthroughActionCfg"

    def __init__(self, cfg: "PassthroughActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, self.cfg.action_dim, device=env.device)

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, action: torch.Tensor):
        self._raw_actions = action.clone()

    def apply_actions(self):
        pass  # applied in HideAndSeekEnv._pre_physics_step


@configclass
class PassthroughActionCfg(ActionTermCfg):
    """Config for the no-op passthrough action term."""

    class_type: type = PassthroughAction
    asset_name: str  = MISSING
    action_dim: int  = ACTION_DIM  # 6 = 3 velocity commands × 2 agents


# ---------------------------------------------------------------------------
# Joint target application helper
# ---------------------------------------------------------------------------

JOINTS_PER_AGENT = 12  # G1 leg DOF count


def apply_joint_targets(
    env: "ManagerBasedRLEnv",
    joint_targets: torch.Tensor,
    agent_name: str,
    joint_ids: "list[int] | None" = None,
) -> None:
    """Apply absolute joint position targets to an agent's articulation.

    Args:
        env: Environment instance.
        joint_targets: Tensor of shape (num_envs, len(joint_ids)).
        agent_name: Name of the agent in the scene ("seeker" or "hider").
        joint_ids: Optional list of joint indices to target. If None, targets all.
    """
    agent = env.scene[agent_name]
    if joint_ids is not None:
        agent.set_joint_position_target(joint_targets, joint_ids=joint_ids)
    else:
        agent.set_joint_position_target(joint_targets)
