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
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from hide_and_seek.env import events, observations, rewards, terminations

# ---------------------------------------------------------------------------
# Load G1 config from unitree_sim_isaaclab (preferred) or isaaclab_assets.
# unitree_sim_isaaclab must be cloned to ~/unitree_sim_isaaclab and
# PROJECT_ROOT env var must point to that directory.
# ---------------------------------------------------------------------------
import sys as _sys
import os as _os

# Add unitree_sim_isaaclab to path so we can import its robot configs
_unitree_path = _os.path.expanduser("~/unitree_sim_isaaclab")
if _os.path.isdir(_unitree_path) and _unitree_path not in _sys.path:
    _sys.path.insert(0, _unitree_path)
    # PROJECT_ROOT tells unitree configs where to find the USD assets
    if "PROJECT_ROOT" not in _os.environ:
        _os.environ["PROJECT_ROOT"] = _unitree_path

try:
    # Import base-fix G1 config (fixed base — good for initial env testing)
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
                "Clone unitree_sim_isaaclab to ~/unitree_sim_isaaclab and run fetch_assets.sh, "
                "or install isaaclab_assets with a G1 config."
            )

_project_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
_replica_scene_usd = _os.environ.get(
    "HNS_REPLICA_SCENE_USD",
    _os.path.join(_project_dir, "data", "scenes", "replica_usd", "apartment_0", "mesh.usd"),
)


# ---------------------------------------------------------------------------
# Scene configuration
# ---------------------------------------------------------------------------
@configclass
class HideAndSeekSceneCfg(InteractiveSceneCfg):
    """Scene with two G1 robots, cameras, ground plane, and lights."""

    # Ground plane kept as a fallback/backup collider.
    ground = sim_utils.GroundPlaneCfg()

    # Replica scene mesh (set HNS_REPLICA_SCENE_USD to override path).
    replica_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReplicaScene",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=_replica_scene_usd,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # Dome light for ambient illumination
    dome_light = sim_utils.DomeLightCfg(
        intensity=1000.0,
        color=(1.0, 1.0, 1.0),
    )

    # Seeker G1 robot
    seeker: ArticulationCfg = _G1_BASE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Seeker",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={".*": 0.0},
        ),
    )

    # Hider G1 robot
    hider: ArticulationCfg = _G1_BASE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Hider",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(5.0, 0.0, 0.8),
            joint_pos={".*": 0.0},
        ),
    )

    # Seeker front camera (matches Unitree G1 d435_link hierarchy)
    seeker_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Seeker/d435_link/front_cam",
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),  # Forward-facing
            convention="ros",
        ),
        height=256,
        width=256,
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.0,
            horizontal_aperture=3.6,
        ),
    )

    # Hider front camera (matches Unitree G1 d435_link hierarchy)
    hider_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Hider/d435_link/front_cam",
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        height=256,
        width=256,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.0,
            horizontal_aperture=3.6,
        ),
    )


# ---------------------------------------------------------------------------
# Observation groups
# ---------------------------------------------------------------------------
@configclass
class SeekerObsCfg(ObservationGroupCfg):
    """Observations for the seeker agent."""

    root_pos = ObservationTermCfg(func=observations.seeker_root_pos)
    root_quat = ObservationTermCfg(func=observations.seeker_root_quat)
    joint_pos = ObservationTermCfg(func=observations.seeker_joint_pos)
    joint_vel = ObservationTermCfg(func=observations.seeker_joint_vel)
    phase = ObservationTermCfg(func=observations.game_phase)
    timer = ObservationTermCfg(func=observations.phase_timer)
    relative_hider = ObservationTermCfg(func=observations.relative_hider_position)


@configclass
class HiderObsCfg(ObservationGroupCfg):
    """Observations for the hider agent."""

    root_pos = ObservationTermCfg(func=observations.hider_root_pos)
    root_quat = ObservationTermCfg(func=observations.hider_root_quat)
    joint_pos = ObservationTermCfg(func=observations.hider_joint_pos)
    joint_vel = ObservationTermCfg(func=observations.hider_joint_vel)
    phase = ObservationTermCfg(func=observations.game_phase)
    timer = ObservationTermCfg(func=observations.phase_timer)


@configclass
class ObservationsCfg:
    """All observation groups."""

    seeker: SeekerObsCfg = SeekerObsCfg()
    hider: HiderObsCfg = HiderObsCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Reward terms (placeholders — will be tuned during training phase)."""

    seeker_detection = RewardTermCfg(
        func=rewards.seeker_detection_reward, weight=10.0
    )
    hider_survival = RewardTermCfg(
        func=rewards.hider_survival_reward, weight=0.1
    )
    seeker_approach = RewardTermCfg(
        func=rewards.seeker_approach_reward, weight=1.0
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    hider_found = TerminationTermCfg(
        func=terminations.hider_detected, time_out=False
    )
    phase_done = TerminationTermCfg(
        func=terminations.game_phase_done, time_out=True
    )
    fallen = TerminationTermCfg(
        func=terminations.out_of_bounds, time_out=False
    )


# ---------------------------------------------------------------------------
# Events (reset randomization)
# ---------------------------------------------------------------------------
@configclass
class EventsCfg:
    """Randomization events applied on episode reset."""

    reset_agents = EventTermCfg(
        func=events.reset_agents_to_random_positions,
        mode="reset",
        params={
            "min_spawn_distance": 3.0,
            "spawn_height": 0.8,
            "spawn_range": 5.0,
        },
    )


# ---------------------------------------------------------------------------
# Top-level environment configuration
# ---------------------------------------------------------------------------
@configclass
class HideAndSeekEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for the hide-and-seek environment."""

    # Scene
    scene: HideAndSeekSceneCfg = HideAndSeekSceneCfg(
        num_envs=1,
        env_spacing=0.0,
    )

    # Managers
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Simulation timing
    sim_dt: float = 1.0 / 60.0  # 60 Hz physics
    decimation: int = 2  # Control at 30 Hz (60/2)
    episode_length_s: float = 40.0  # 10s hiding + 30s seeking

    # --- Game-specific parameters ---
    hiding_phase_steps: int = 150  # 5 sec at 30 Hz control
    seeking_phase_steps: int = 900  # 30 sec at 30 Hz control
    min_spawn_distance: float = 3.0  # meters
    detection_pixel_threshold: float = 0.005  # 0.5% of pixels
    detection_confirmation_steps: int = 3  # consecutive frames
