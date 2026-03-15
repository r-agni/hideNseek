"""Convert Replica Dataset scenes from PLY to USD for Isaac Lab.

Usage:
    python scripts/convert_replica.py --scene apartment_0
    python scripts/convert_replica.py --all
    python scripts/convert_replica.py --scene-dir data/scenes/replica_raw/apartment_0 \
        --output-dir data/scenes/replica_usd/apartment_0
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hide_and_seek.utils.scene_converter import convert_replica_scene

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "scenes", "replica_raw")
USD_DIR = os.path.join(PROJECT_DIR, "data", "scenes", "replica_usd")


def main():
    parser = argparse.ArgumentParser(description="Convert Replica scenes to USD")
    parser.add_argument("--scene", type=str, default="apartment_0", help="Scene name to convert")
    parser.add_argument("--scene-dir", type=str, default=None, help="Explicit path to a scene directory")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=USD_DIR,
        help="Output directory (parent or scene-specific directory)",
    )
    parser.add_argument("--all", action="store_true", help="Convert all available scenes")
    args = parser.parse_args()

    if args.scene_dir and args.all:
        log.error("Use either --scene-dir or --all, not both.")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)

    if args.scene_dir:
        scene_dir = os.path.abspath(args.scene_dir)
        if not os.path.isdir(scene_dir):
            log.error(f"Scene directory not found: {scene_dir}")
            sys.exit(1)
        scene_pairs = [(os.path.basename(os.path.normpath(scene_dir)), scene_dir)]
    elif args.all:
        if not os.path.isdir(RAW_DIR):
            log.error(f"Raw scenes directory not found: {RAW_DIR}")
            log.error("Run 'bash scripts/setup_data.sh' first.")
            sys.exit(1)
        scene_pairs = [
            (d, os.path.join(RAW_DIR, d))
            for d in os.listdir(RAW_DIR)
            if os.path.isdir(os.path.join(RAW_DIR, d))
        ]
    else:
        scene_pairs = [(args.scene, os.path.join(RAW_DIR, args.scene))]

    for scene_name, scene_dir in scene_pairs:
        if not os.path.isdir(scene_dir):
            log.warning(f"Scene directory not found: {scene_dir}, skipping.")
            continue

        log.info(f"Converting scene: {scene_name}")
        try:
            usd_path = convert_replica_scene(scene_dir, output_dir)
            log.info(f"Success: {usd_path}")
        except Exception as e:
            log.error(f"Failed to convert {scene_name}: {e}")
            raise


if __name__ == "__main__":
    main()
