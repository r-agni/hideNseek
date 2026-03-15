"""Replica scene asset configurations for Isaac Lab."""

import os

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

# Project root for resolving relative data paths
_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@configclass
class ReplicaSceneCfg:
    """Configuration for a Replica Dataset scene loaded as a static USD mesh."""

    usd_path: str = os.path.join(
        _PROJECT_DIR, "data", "scenes", "replica_usd", "apartment_0", "mesh.usd"
    )

    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    )
