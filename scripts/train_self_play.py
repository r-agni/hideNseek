"""Run seeker-vs-hider self-play rollouts.

Examples:
  conda run -n hide-and-seek python scripts/train_self_play.py --headless
  conda run -n hide-and-seek python scripts/train_self_play.py --headless --hider-llm
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hide_and_seek.training import (
    LLMPolicyConfig,
    RandomVelocityPolicy,
    SelfPlayConfig,
    SelfPlayRunner,
)
from hide_and_seek.training.self_play import LLMVelocityPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hide-and-seek self-play rollouts")
    parser.add_argument("--headless", action="store_true", help="Run Isaac in headless mode")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs")
    parser.add_argument("--steps", type=int, default=2000, help="Total rollout steps")
    parser.add_argument("--log-every", type=int, default=100, help="Logging interval")
    parser.add_argument("--seeker-llm", action="store_true", help="Use LLM policy for seeker")
    parser.add_argument("--hider-llm", action="store_true", help="Use LLM policy for hider")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="google/gemma-3-27b-it-fast",
        help="OpenAI-compatible model id",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="https://api.tokenfactory.nebius.com/v1/",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--llm-key-env",
        type=str,
        default="NEBIUS_API_KEY",
        help="Environment variable name for API key",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app

    from hide_and_seek.env.hide_and_seek_env import HideAndSeekEnv
    from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg

    cfg = HideAndSeekEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = HideAndSeekEnv(cfg)

    llm_cfg = LLMPolicyConfig(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key_env=args.llm_key_env,
    )

    seeker_policy = LLMVelocityPolicy(llm_cfg) if args.seeker_llm else RandomVelocityPolicy()
    hider_policy = LLMVelocityPolicy(llm_cfg) if args.hider_llm else RandomVelocityPolicy()

    runner = SelfPlayRunner(
        env=env,
        cfg=SelfPlayConfig(total_steps=args.steps, log_every=args.log_every),
        seeker_policy=seeker_policy,
        hider_policy=hider_policy,
    )

    metrics = runner.run()
    print("[self_play] final_metrics:", metrics)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
