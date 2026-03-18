"""Inspect the G1 base config to see actuator keys and init state."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from isaaclab_assets import G1_MINIMAL_CFG as cfg
    print("Using G1_MINIMAL_CFG")
except ImportError:
    from isaaclab_assets import G1_CFG as cfg
    print("Using G1_CFG")

print("Actuator keys:", list(cfg.actuators.keys()))
for k, v in cfg.actuators.items():
    print(f"  {k}: stiffness={v.stiffness}, damping={v.damping}, joint_names_expr={v.joint_names_expr}")
print("Init state joint_pos:", cfg.init_state.joint_pos)
print("Init state pos:", cfg.init_state.pos)
