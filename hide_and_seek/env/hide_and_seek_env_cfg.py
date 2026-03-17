"""Environment configuration for hide-and-seek with two Unitree G1 robots."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from hide_and_seek.env import events, observations, rewards, terminations
from hide_and_seek.env.actions import ACTION_DIM, PassthroughActionCfg

# ---------------------------------------------------------------------------
# G1 robot config â€” prefer unitree_sim_isaaclab, fall back to isaaclab_assets
# ---------------------------------------------------------------------------
import sys as _sys
import os as _os

# Optional environment scene USD (e.g. a Nucleus scene with walls/rooms).
# Leave empty to use flat ground only (fastest, no network streaming).
# Set HIDE_AND_SEEK_SCENE_USD env var to load a specific USD at runtime.
_SCENE_USD = _os.environ.get("HIDE_AND_SEEK_SCENE_USD", "")
_SCENE_Z_OFFSET = 0.0

_unitree_path = _os.path.expanduser("~/unitree_sim_isaaclab")
if _os.path.isdir(_unitree_path) and _unitree_path not in _sys.path:
    _sys.path.insert(0, _unitree_path)
    if "PROJECT_ROOT" not in _os.environ:
        _os.environ["PROJECT_ROOT"] = _unitree_path

try:
    from robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX as _G1_BASE_CFG
except ImportError:
    try:
        from isaaclab_assets import G1_MINIMAL_CFG as _G1_BASE_CFG
    except ImportError:
        try:
            from isaaclab_assets import G1_CFG as _G1_BASE_CFG
        except ImportError:
            raise ImportError(
                "Could not import G1 robot config. "
                "Clone unitree_sim_isaaclab to ~/unitree_sim_isaaclab or "
                "install isaaclab_assets with a G1 config."
            )


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
@configclass
class HideAndSeekSceneCfg(InteractiveSceneCfg):
    """Flat-ground arena with two G1 robots and an overview camera."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    seeker: ArticulationCfg = _G1_BASE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Seeker",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-1.5, 0.0, 0.80),
            # Use G1_CFG standing pose: hip_pitch=-0.20, knee=0.42, ankle=-0.23, etc.
        ),
        actuators={**_G1_BASE_CFG.actuators, "arms": _G1_BASE_CFG.actuators["arms"].replace(stiffness=0.0, damping=5.0)},
    )

    hider: ArticulationCfg = _G1_BASE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Hider",
        spawn=_G1_BASE_CFG.spawn.replace(
            semantic_tags=[("class", "hider")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.80),
            # Use G1_CFG standing pose: hip_pitch=-0.20, knee=0.42, ankle=-0.23, etc.
        ),
        actuators={**_G1_BASE_CFG.actuators, "arms": _G1_BASE_CFG.actuators["arms"].replace(stiffness=0.0, damping=5.0)},
    )

    # Cameras disabled — detection uses proximity (≤0.5 m) instead of vision.
    # Re-enable for demo/visualization only (adds significant rendering overhead).
    # seeker_camera: CameraCfg = CameraCfg(...)

    # Overhead camera disabled during training — renders 1280×720 RGB every step
    # which consumes ~80% of GPU time. Re-enable for visualization only.
    # overhead_camera: CameraCfg = CameraCfg(...)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """6D velocity commands (3 per agent) â€” decoded to joint targets in env."""

    velocity = PassthroughActionCfg(asset_name="seeker", action_dim=ACTION_DIM)


# ---------------------------------------------------------------------------
# Observation groups
# ---------------------------------------------------------------------------
@configclass
class SeekerObsCfg(ObservationGroupCfg):
    """Observations for the seeker agent."""

    root_pos  = ObservationTermCfg(func=observations.seeker_root_pos)
    root_quat = ObservationTermCfg(func=observations.seeker_root_quat)
    lin_vel   = ObservationTermCfg(func=observations.seeker_lin_vel)
    ang_vel   = ObservationTermCfg(func=observations.seeker_ang_vel)
    joint_pos = ObservationTermCfg(func=observations.seeker_joint_pos)
    joint_vel = ObservationTermCfg(func=observations.seeker_joint_vel)
    phase     = ObservationTermCfg(func=observations.game_phase)
    timer     = ObservationTermCfg(func=observations.phase_timer)
    detection = ObservationTermCfg(func=observations.seeker_detection_signal)


@configclass
class HiderObsCfg(ObservationGroupCfg):
    """Observations for the hider agent."""

    root_pos  = ObservationTermCfg(func=observations.hider_root_pos)
    root_quat = ObservationTermCfg(func=observations.hider_root_quat)
    lin_vel   = ObservationTermCfg(func=observations.hider_lin_vel)
    ang_vel   = ObservationTermCfg(func=observations.hider_ang_vel)
    joint_pos = ObservationTermCfg(func=observations.hider_joint_pos)
    joint_vel = ObservationTermCfg(func=observations.hider_joint_vel)
    phase     = ObservationTermCfg(func=observations.game_phase)
    timer     = ObservationTermCfg(func=observations.phase_timer)


@configclass
class ObservationsCfg:
    seeker: SeekerObsCfg = SeekerObsCfg()
    hider:  HiderObsCfg  = HiderObsCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    # Seeker — proximity-based: reach the hider
    seeker_detection  = RewardTermCfg(func=rewards.seeker_detection_reward, weight=10.0)
    seeker_approach   = RewardTermCfg(func=rewards.seeker_approach_reward,  weight=2.0)
    seeker_alive      = RewardTermCfg(func=rewards.seeker_alive_bonus,      weight=0.01)
    # Hider — stay far, move to cover during hiding phase
    hider_survival        = RewardTermCfg(func=rewards.hider_survival_reward,        weight=1.0)
    hider_distance        = RewardTermCfg(func=rewards.hider_distance_reward,        weight=0.5)
    hider_hiding_movement = RewardTermCfg(func=rewards.hider_hiding_movement_reward, weight=0.2)
    hider_alive           = RewardTermCfg(func=rewards.hider_alive_bonus,            weight=0.01)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    hider_found  = TerminationTermCfg(func=terminations.hider_detected,   time_out=False)
    phase_done   = TerminationTermCfg(func=terminations.game_phase_done,  time_out=True)
    robot_fallen = TerminationTermCfg(func=terminations.robot_fallen,     time_out=False)


# ---------------------------------------------------------------------------
# Events (reset randomization)
# ---------------------------------------------------------------------------
@configclass
class EventsCfg:
    reset_agents = EventTermCfg(
        func=events.reset_agents_to_random_positions,
        mode="reset",
        params={
            "min_spawn_distance": 2.0,
            "spawn_height": 0.80,  # G1 pelvis ~0.74m; 0.80 gives clearance for PhysX to settle
            "spawn_range": 3.0,    # hospital corridors ~6m wide; stay away from perimeter walls
        },
    )


# ---------------------------------------------------------------------------
# Top-level environment configuration
# ---------------------------------------------------------------------------
@configclass
class HideAndSeekEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for the hide-and-seek environment."""

    scene:        HideAndSeekSceneCfg = HideAndSeekSceneCfg(num_envs=1, env_spacing=0.0)
    actions:      ActionsCfg          = ActionsCfg()
    observations: ObservationsCfg     = ObservationsCfg()
    rewards:      RewardsCfg          = RewardsCfg()
    terminations: TerminationsCfg     = TerminationsCfg()
    events:       EventsCfg           = EventsCfg()

    # Simulation timing
    sim_dt:           float = 1.0 / 60.0  # 60 Hz physics
    decimation:       int   = 2            # control at 30 Hz
    episode_length_s: float = 45.0         # 15s hiding + 30s seeking

    # Game parameters
    hiding_phase_steps:           int   = 450    # 15s at 30 Hz
    seeking_phase_steps:          int   = 900    # 30s at 30 Hz
    min_spawn_distance:           float = 3.0    # meters
    detection_confirmation_steps: int   = 3      # consecutive frames

    # Path to the pretrained G1 locomotion policy (motion.pt).
    # Set this to ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt or
    # pass via train_rl.py --policy-path argument.
    locomotion_policy_path: str = ""



