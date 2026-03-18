"""Convert g1_12dof.urdf to USD for use in Isaac Lab.

Run once — takes ~5-10 min depending on mesh complexity.

Usage:
    conda run -n hide-and-seek python scripts/convert_g1_12dof.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from pathlib import Path

src = str(Path.home() / "unitree_rl_gym" / "resources" / "robots" / "g1_description" / "g1_12dof.urdf")
out_dir = str(Path(__file__).resolve().parents[1] / "data" / "models" / "g1_12dof")

print(f"[convert_g1_12dof] input:  {src}")
print(f"[convert_g1_12dof] output: {out_dir}")

cfg = UrdfConverterCfg(
    asset_path=src,
    usd_dir=out_dir,
    force_usd_conversion=True,
    fix_base=False,
    make_instanceable=False,
    joint_drive=None,  # drive gains set by ImplicitActuator in training, not the USD
)
converter = UrdfConverter(cfg)
print(f"[convert_g1_12dof] done → {converter.usd_path}")

launcher.app.close()
