"""Convert hospital.usd to a version with baked physics collision meshes.

Run once — takes ~10-30 min depending on mesh complexity.

Usage:
    conda run -n hide-and-seek python scripts/convert_hospital.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)

from isaaclab.sim.converters import UsdConverter, UsdConverterCfg
from pathlib import Path

src = str(Path(__file__).resolve().parents[1] / "data" / "scenes" / "hospital" / "hospital.usd")
out_dir = str(Path(__file__).resolve().parents[1] / "data" / "scenes" / "hospital_collision")

print(f"[convert_hospital] input:  {src}")
print(f"[convert_hospital] output: {out_dir}")

cfg = UsdConverterCfg(
    asset_path=src,
    usd_dir=out_dir,
    force_usd_conversion=True,
    make_instanceable=False,
    collision_approximation="convexDecomposition",
)
converter = UsdConverter(cfg)
print(f"[convert_hospital] done → {converter.usd_path}")

launcher.app.close()
