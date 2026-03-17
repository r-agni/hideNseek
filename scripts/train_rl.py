"""PPO RL training for hide-and-seek with pretrained G1 locomotion policy.

The RL policy learns to output 6D velocity commands
    [seeker_vx, seeker_vy, seeker_yaw, hider_vx, hider_vy, hider_yaw]
which are fed into the frozen pretrained locomotion policy (motion.pt)
that converts them to 12 joint position targets per agent.

Examples:
  # Interactive training (opens Isaac Sim window, 1 env):
  conda run -n hide-and-seek python scripts/train_rl.py \\
      --policy-path ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt

  # Headless training (no window, faster):
  conda run -n hide-and-seek python scripts/train_rl.py --headless \\
      --policy-path ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt

  # Resume from checkpoint:
  conda run -n hide-and-seek python scripts/train_rl.py \\
      --policy-path ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt \\
      --checkpoint runs/checkpoints/policy_000100000.pt

  # Short run with video recording:
  conda run -n hide-and-seek python scripts/train_rl.py --headless \\
      --steps 300 --record \\
      --policy-path ~/unitree_rl_gym/deploy/pre_train/g1/motion.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _default_policy_path() -> str:
    candidates = [
        Path.home() / "unitree_rl_gym" / "deploy" / "pre_train" / "g1" / "motion.pt",
        Path(__file__).resolve().parents[1] / "data" / "models" / "g1_motion.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hide-and-seek PPO RL training")
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--num-envs",   type=int, default=1)
    parser.add_argument("--steps",      type=int, default=5_000_000)
    parser.add_argument(
        "--policy-path",
        type=str,
        default=_default_policy_path(),
        help="Path to pretrained G1 motion.pt (TorchScript JIT).",
    )
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Resume from checkpoint .pt file.")
    parser.add_argument("--rollout-steps",  type=int,   default=2048)
    parser.add_argument("--num-epochs",     type=int,   default=10)
    parser.add_argument("--minibatch-size", type=int,   default=512)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str,   default="runs/checkpoints")
    parser.add_argument("--record",     action="store_true")
    parser.add_argument("--video-path", type=str, default="runs/recording.mp4")
    parser.add_argument(
        "--kit-args",
        type=str,
        default="",
        help="Extra Omniverse Kit CLI args (passed through AppLauncher).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher
    # No cameras in scene — proximity-based detection. Disabling saves ~80% GPU time.
    launcher = AppLauncher(
        headless=args.headless,
        enable_cameras=False,
        kit_args=args.kit_args,
    )

    from hide_and_seek.env.hide_and_seek_env import HideAndSeekEnv
    from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg
    from hide_and_seek.training.rl_runner import RLConfig, RLRunner

    cfg = HideAndSeekEnvCfg()
    cfg.scene.num_envs          = args.num_envs
    cfg.locomotion_policy_path  = args.policy_path

    env = HideAndSeekEnv(cfg)
    if not args.headless:
        # Force a visible default editor camera so the GUI doesn't open on a black view.
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=[0.0, -8.0, 2.5], target=[0.0, 0.0, 0.9])
        except Exception as exc:
            print(f"[train_rl] warning: could not set viewport camera: {exc}")

    rl_cfg = RLConfig(
        total_steps    = args.steps,
        rollout_steps  = args.rollout_steps,
        num_epochs     = args.num_epochs,
        minibatch_size = args.minibatch_size,
        lr             = args.lr,
        checkpoint_dir = args.checkpoint_dir,
        record         = args.record,
        video_path     = args.video_path,
    )

    runner = RLRunner(env, rl_cfg)

    if args.checkpoint:
        runner.load_checkpoint(args.checkpoint)

    runner.run()

    env.close()
    launcher.app.close()


if __name__ == "__main__":
    main()
