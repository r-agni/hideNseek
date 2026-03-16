"""Self-play training utilities for hide-and-seek.

Includes:
- random velocity baseline policies
- optional OpenAI-compatible LLM velocity policy (for VLA/VLM-style control)
- simple self-play runner that rolls the environment and logs metrics
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import torch


@dataclass
class SelfPlayConfig:
    """Configuration for the self-play rollout loop."""

    total_steps: int = 5_000
    log_every: int = 100
    reset_on_done: bool = True


@dataclass
class LLMPolicyConfig:
    """Configuration for optional LLM action generation."""

    base_url: str = "https://api.tokenfactory.nebius.com/v1/"
    model: str = "google/gemma-3-27b-it-fast"
    api_key_env: str = "NEBIUS_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 128
    system_prompt: str = (
        "You control a humanoid robot in hide-and-seek. "
        "Return ONLY compact JSON: {\"linear\": float, \"angular\": float} "
        "with each in [-1, 1]."
    )


class VelocityPolicy:
    """Interface for policies that output normalized base velocity actions."""

    def act(self, obs_group: dict[str, torch.Tensor], role: str, num_envs: int, device: torch.device) -> torch.Tensor:
        raise NotImplementedError


class RandomVelocityPolicy(VelocityPolicy):
    """Uniform random baseline policy."""

    def act(self, obs_group: dict[str, torch.Tensor], role: str, num_envs: int, device: torch.device) -> torch.Tensor:
        return torch.rand(num_envs, 2, device=device) * 2.0 - 1.0


class LLMVelocityPolicy(VelocityPolicy):
    """OpenAI-compatible LLM policy that maps observations to velocity actions.

    This is intended as a research scaffold for VLM/VLA-style controllers.
    """

    def __init__(self, cfg: LLMPolicyConfig):
        self.cfg = cfg
        self._enabled = False
        self._client = None

        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            return

        try:
            from openai import OpenAI
        except ImportError:
            return

        self._client = OpenAI(base_url=cfg.base_url, api_key=api_key)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def act(self, obs_group: dict[str, torch.Tensor], role: str, num_envs: int, device: torch.device) -> torch.Tensor:
        if not self._enabled:
            return torch.zeros(num_envs, 2, device=device)

        actions = []
        for env_idx in range(num_envs):
            user_prompt = self._build_prompt(obs_group, role, env_idx)
            try:
                response = self._client.chat.completions.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    messages=[
                        {"role": "system", "content": self.cfg.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
            except Exception:
                content = ""
            actions.append(self._parse_action(content))

        return torch.tensor(actions, dtype=torch.float32, device=device).clamp(-1.0, 1.0)

    def _build_prompt(self, obs_group: dict[str, torch.Tensor], role: str, env_idx: int) -> str:
        obs = {}
        for key, value in obs_group.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                env_val = value[env_idx].detach().float().cpu().flatten().tolist()
                obs[key] = [round(float(v), 4) for v in env_val[:16]]
        return (
            f"Role: {role}\n"
            "Choose motion for this step.\n"
            f"Observations (truncated): {json.dumps(obs, separators=(',', ':'))}\n"
            "Output JSON only."
        )

    @staticmethod
    def _parse_action(content: str) -> list[float]:
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return [0.0, 0.0]
            payload = json.loads(content[start : end + 1])
            linear = float(payload.get("linear", 0.0))
            angular = float(payload.get("angular", 0.0))
            return [linear, angular]
        except Exception:
            return [0.0, 0.0]


class SelfPlayRunner:
    """Runs a two-policy seeker/hider self-play loop in the Isaac env."""

    def __init__(
        self,
        env: Any,
        cfg: SelfPlayConfig,
        seeker_policy: VelocityPolicy,
        hider_policy: VelocityPolicy,
    ):
        self.env = env
        self.cfg = cfg
        self.seeker_policy = seeker_policy
        self.hider_policy = hider_policy

    def run(self) -> dict[str, float]:
        obs, _ = self.env.reset()
        detected_events = 0
        done_events = 0

        for step in range(1, self.cfg.total_steps + 1):
            seeker_obs = obs.get("seeker", {})
            hider_obs = obs.get("hider", {})

            seeker_action = self.seeker_policy.act(
                seeker_obs, role="seeker", num_envs=self.env.num_envs, device=self.env.device
            )
            hider_action = self.hider_policy.act(
                hider_obs, role="hider", num_envs=self.env.num_envs, device=self.env.device
            )

            action = torch.cat([seeker_action, hider_action], dim=-1)
            obs, _, terminated, truncated, _ = self.env.step(action)

            if hasattr(self.env, "hider_detected"):
                detected_events += int(self.env.hider_detected.sum().item())

            done_mask = terminated | truncated
            if done_mask.any():
                done_events += int(done_mask.sum().item())
                if self.cfg.reset_on_done:
                    obs, _ = self.env.reset()

            if step % self.cfg.log_every == 0:
                print(
                    f"[self_play] step={step} "
                    f"detected_events={detected_events} done_events={done_events}"
                )

        return {
            "steps": float(self.cfg.total_steps),
            "detected_events": float(detected_events),
            "done_events": float(done_events),
        }
