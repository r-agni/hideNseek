"""PPO RL training loop for hide-and-seek.

Architecture:
  - ActorCritic: separate MLP heads for seeker and hider.
    Each head outputs a 3D Gaussian policy (vx, vy, yaw_rate) and a value.
  - RLRunner: collects rollouts from the env, computes independent GAE
    returns/advantages for each agent, and runs PPO with separate losses.

The env's action space is 6D: [seeker_vx, seeker_vy, seeker_yaw,
                                 hider_vx,  hider_vy,  hider_yaw]
All ∈ [-1, 1].  The env scales these to physical units internally.

Key design: seeker and hider have SEPARATE reward streams, separate GAE,
and independent PPO clipping. This is critical because their rewards are
negatively correlated — blending them produces near-zero advantages and
vanishing policy gradients.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RLConfig:
    """Hyperparameters for PPO training."""
    total_steps:    int   = 5_000_000
    rollout_steps:  int   = 2048        # steps collected per env per update
    num_epochs:     int   = 10          # PPO epochs per rollout
    minibatch_size: int   = 512
    lr:             float = 3e-4
    gamma:          float = 0.99
    gae_lambda:     float = 0.95
    clip_eps:       float = 0.2         # PPO clipping epsilon
    value_coef:     float = 0.5
    entropy_coef:   float = 0.01
    max_grad_norm:  float = 0.5
    log_every:      int   = 512         # env steps between console logs
    save_every:     int   = 10_000      # env steps between checkpoints
    checkpoint_dir: str   = "runs/checkpoints"
    record:         bool  = False
    record_every:   int   = 5           # physics steps between frames
    video_fps:      int   = 10
    video_path:     str   = "runs/recording.mp4"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Simple 3-layer MLP with Tanh activations."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Separate actor-critic heads for seeker and hider.

    Each agent has:
      - Actor: obs → mean of 3D Gaussian (vx, vy, yaw).
               log_std is a learnable parameter (not input-dependent).
      - Critic: obs → scalar value estimate.

    The two agents share no weights — they learn independent strategies.
    Critics are trained on independent reward streams (seeker vs. hider),
    so blending their values would corrupt both GAE computations.
    """

    HIDDEN = 256

    def __init__(self, seeker_obs_dim: int, hider_obs_dim: int):
        super().__init__()

        # Seeker
        self.seeker_actor   = _MLP(seeker_obs_dim, self.HIDDEN, 3)
        self.seeker_critic  = _MLP(seeker_obs_dim, self.HIDDEN, 1)
        self.seeker_log_std = nn.Parameter(torch.zeros(3))

        # Hider
        self.hider_actor   = _MLP(hider_obs_dim, self.HIDDEN, 3)
        self.hider_critic  = _MLP(hider_obs_dim, self.HIDDEN, 1)
        self.hider_log_std = nn.Parameter(torch.zeros(3))

        # Orthogonal init for actor output layers (recommended for PPO)
        for actor in (self.seeker_actor, self.hider_actor):
            nn.init.orthogonal_(actor.net[-1].weight, gain=0.01)
            nn.init.zeros_(actor.net[-1].bias)

    def get_seeker_dist(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.seeker_actor(obs))
        std  = self.seeker_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_hider_dist(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.hider_actor(obs))
        std  = self.hider_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def seeker_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.seeker_critic(obs).squeeze(-1)

    def hider_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.hider_critic(obs).squeeze(-1)

    def act(self, seeker_obs: torch.Tensor, hider_obs: torch.Tensor):
        """Sample actions and compute per-agent log-probs.

        Returns:
            actions:    (N, 6) clipped to [-1, 1]
            s_lp:       (N,)   seeker log-prob
            h_lp:       (N,)   hider log-prob
            seeker_val: (N,)
            hider_val:  (N,)
        """
        s_dist = self.get_seeker_dist(seeker_obs)
        h_dist = self.get_hider_dist(hider_obs)

        s_action = s_dist.sample().clamp(-1.0, 1.0)
        h_action = h_dist.sample().clamp(-1.0, 1.0)

        s_lp = s_dist.log_prob(s_action).sum(-1)
        h_lp = h_dist.log_prob(h_action).sum(-1)

        actions = torch.cat([s_action, h_action], dim=-1)
        return actions, s_lp, h_lp, self.seeker_value(seeker_obs), self.hider_value(hider_obs)

    def evaluate(
        self,
        seeker_obs: torch.Tensor,
        hider_obs: torch.Tensor,
        actions: torch.Tensor,
    ):
        """Evaluate log-probs and entropy for stored actions (PPO update).

        Returns:
            s_lp:       (N,)
            h_lp:       (N,)
            s_ent:      (N,)
            h_ent:      (N,)
            seeker_val: (N,)
            hider_val:  (N,)
        """
        s_dist = self.get_seeker_dist(seeker_obs)
        h_dist = self.get_hider_dist(hider_obs)

        s_action = actions[:, 0:3]
        h_action = actions[:, 3:6]

        s_lp  = s_dist.log_prob(s_action).sum(-1)
        h_lp  = h_dist.log_prob(h_action).sum(-1)
        s_ent = s_dist.entropy().sum(-1)
        h_ent = h_dist.entropy().sum(-1)

        return s_lp, h_lp, s_ent, h_ent, self.seeker_value(seeker_obs), self.hider_value(hider_obs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class RLRunner:
    """PPO training loop with separate reward streams for seeker and hider."""

    def __init__(self, env: Any, cfg: RLConfig):
        self.env = env
        self.cfg = cfg
        self.device   = env.device
        self.num_envs = env.num_envs

        # Resolve observation dimensions from the env after reset
        obs, _ = env.reset()
        seeker_obs_dim = self._flat_obs(obs, "seeker").shape[-1]
        hider_obs_dim  = self._flat_obs(obs, "hider").shape[-1]
        print(f"[rl_runner] seeker_obs_dim={seeker_obs_dim}  hider_obs_dim={hider_obs_dim}")

        self.policy    = ActorCritic(seeker_obs_dim, hider_obs_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)

        # Rollout buffer — separate reward and log_prob streams per agent
        R = cfg.rollout_steps
        N = self.num_envs
        self._buf = {
            "seeker_obs":  torch.zeros(R, N, seeker_obs_dim, device=self.device),
            "hider_obs":   torch.zeros(R, N, hider_obs_dim,  device=self.device),
            "actions":     torch.zeros(R, N, 6,               device=self.device),
            "s_log_probs": torch.zeros(R, N,                  device=self.device),
            "h_log_probs": torch.zeros(R, N,                  device=self.device),
            "s_rewards":   torch.zeros(R, N,                  device=self.device),
            "h_rewards":   torch.zeros(R, N,                  device=self.device),
            "dones":       torch.zeros(R, N,                  device=self.device),
            "s_values":    torch.zeros(R, N,                  device=self.device),
            "h_values":    torch.zeros(R, N,                  device=self.device),
        }
        self._last_obs = obs
        self._frames: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        """Main training loop."""
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        total_env_steps = 0
        t_start         = time.time()
        s_ep_rewards: list[float] = []
        h_ep_rewards: list[float] = []

        print(f"[rl_runner] starting PPO training for {self.cfg.total_steps:,} env steps")
        print(f"[rl_runner] {self.num_envs} envs × {self.cfg.rollout_steps} steps = "
              f"{self.num_envs * self.cfg.rollout_steps} transitions/update")

        while total_env_steps < self.cfg.total_steps:
            # --- Collect rollout ---
            s_ep, h_ep = self._collect_rollout()
            s_ep_rewards.extend(s_ep)
            h_ep_rewards.extend(h_ep)
            total_env_steps += self.cfg.rollout_steps * self.num_envs

            # --- PPO update ---
            metrics = self._ppo_update()

            # --- Logging ---
            if total_env_steps % self.cfg.log_every < self.cfg.rollout_steps * self.num_envs:
                elapsed = time.time() - t_start
                fps     = total_env_steps / max(elapsed, 1.0)
                mean_s  = sum(s_ep_rewards[-100:]) / max(len(s_ep_rewards[-100:]), 1)
                mean_h  = sum(h_ep_rewards[-100:]) / max(len(h_ep_rewards[-100:]), 1)
                print(
                    f"[rl_runner] steps={total_env_steps:>9,}  "
                    f"fps={fps:>6.0f}  "
                    f"seeker_rew={mean_s:>7.3f}  "
                    f"hider_rew={mean_h:>7.3f}  "
                    f"s_pol_loss={metrics['s_policy_loss']:>7.4f}  "
                    f"h_pol_loss={metrics['h_policy_loss']:>7.4f}  "
                    f"s_val_loss={metrics['s_value_loss']:>7.4f}  "
                    f"h_val_loss={metrics['h_value_loss']:>7.4f}  "
                    f"entropy={metrics['entropy']:>6.4f}"
                )

            # --- Checkpoint ---
            if total_env_steps % self.cfg.save_every < self.cfg.rollout_steps * self.num_envs:
                self._save_checkpoint(total_env_steps)

        # Final checkpoint + optional video
        self._save_checkpoint(total_env_steps)
        if self.cfg.record and self._frames:
            self._write_video()

        print(f"[rl_runner] training complete — {total_env_steps:,} steps in "
              f"{time.time() - t_start:.0f}s")

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self) -> tuple[list[float], list[float]]:
        """Fill the rollout buffer. Returns (seeker_ep_rewards, hider_ep_rewards)."""
        s_ep_rewards: list[float] = []
        h_ep_rewards: list[float] = []
        s_ep_buf = torch.zeros(self.num_envs, device=self.device)
        h_ep_buf = torch.zeros(self.num_envs, device=self.device)

        self.policy.eval()
        with torch.no_grad():
            for step in range(self.cfg.rollout_steps):
                obs = self._last_obs
                s_obs = self._flat_obs(obs, "seeker")
                h_obs = self._flat_obs(obs, "hider")

                actions, s_lp, h_lp, s_val, h_val = self.policy.act(s_obs, h_obs)

                next_obs, _combined_rew, terminated, truncated, _ = self.env.step(actions)
                dones = (terminated | truncated).float()

                # Read per-agent rewards cached by the env in _post_physics_step
                s_rew = self.env.seeker_step_reward.clone()
                h_rew = self.env.hider_step_reward.clone()

                self._buf["seeker_obs"][step]  = s_obs
                self._buf["hider_obs"][step]   = h_obs
                self._buf["actions"][step]     = actions
                self._buf["s_log_probs"][step] = s_lp
                self._buf["h_log_probs"][step] = h_lp
                self._buf["s_rewards"][step]   = s_rew
                self._buf["h_rewards"][step]   = h_rew
                self._buf["dones"][step]       = dones
                self._buf["s_values"][step]    = s_val
                self._buf["h_values"][step]    = h_val

                s_ep_buf += s_rew
                h_ep_buf += h_rew
                done_mask = dones.bool()
                if done_mask.any():
                    s_ep_rewards.extend(s_ep_buf[done_mask].cpu().tolist())
                    h_ep_rewards.extend(h_ep_buf[done_mask].cpu().tolist())
                    s_ep_buf[done_mask] = 0.0
                    h_ep_buf[done_mask] = 0.0

                self._last_obs = next_obs

                if self.cfg.record and step % self.cfg.record_every == 0:
                    frame = self._grab_camera_frame()
                    if frame is not None:
                        self._frames.append(frame)

        return s_ep_rewards, h_ep_rewards

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict[str, float]:
        """Compute independent GAE for each agent, then run PPO minibatch updates."""
        cfg = self.cfg
        R   = cfg.rollout_steps
        N   = self.num_envs

        # Bootstrap values for last state
        obs = self._last_obs
        with torch.no_grad():
            s_next_val = self.policy.seeker_value(self._flat_obs(obs, "seeker"))
            h_next_val = self.policy.hider_value( self._flat_obs(obs, "hider"))

        s_rewards = self._buf["s_rewards"]   # (R, N)
        h_rewards = self._buf["h_rewards"]
        s_values  = self._buf["s_values"]
        h_values  = self._buf["h_values"]
        dones     = self._buf["dones"]

        # --- Seeker GAE ---
        s_advantages = torch.zeros_like(s_rewards)
        last_s_gae   = torch.zeros(N, device=self.device)
        for t in reversed(range(R)):
            nxt_s_val = s_next_val if t == R - 1 else s_values[t + 1]
            delta     = s_rewards[t] + cfg.gamma * nxt_s_val * (1 - dones[t]) - s_values[t]
            last_s_gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * last_s_gae
            s_advantages[t] = last_s_gae
        s_returns     = s_advantages + s_values
        s_advantages  = (s_advantages - s_advantages.mean()) / (s_advantages.std() + 1e-8)

        # --- Hider GAE ---
        h_advantages = torch.zeros_like(h_rewards)
        last_h_gae   = torch.zeros(N, device=self.device)
        for t in reversed(range(R)):
            nxt_h_val = h_next_val if t == R - 1 else h_values[t + 1]
            delta     = h_rewards[t] + cfg.gamma * nxt_h_val * (1 - dones[t]) - h_values[t]
            last_h_gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * last_h_gae
            h_advantages[t] = last_h_gae
        h_returns     = h_advantages + h_values
        h_advantages  = (h_advantages - h_advantages.mean()) / (h_advantages.std() + 1e-8)

        # Flatten (R × N) → (R*N)
        flat = lambda t: t.reshape(-1, *t.shape[2:])
        b_s_obs    = flat(self._buf["seeker_obs"])
        b_h_obs    = flat(self._buf["hider_obs"])
        b_act      = flat(self._buf["actions"])
        b_s_lp_old = flat(self._buf["s_log_probs"])
        b_h_lp_old = flat(self._buf["h_log_probs"])
        b_s_adv    = flat(s_advantages)
        b_h_adv    = flat(h_advantages)
        b_s_ret    = flat(s_returns)
        b_h_ret    = flat(h_returns)

        total = R * N
        s_pol_losses, h_pol_losses = [], []
        s_val_losses, h_val_losses = [], []
        entropies = []

        self.policy.train()
        for _ in range(cfg.num_epochs):
            idx = torch.randperm(total, device=self.device)
            for start in range(0, total, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]

                s_lp, h_lp, s_ent, h_ent, s_val_mb, h_val_mb = self.policy.evaluate(
                    b_s_obs[mb], b_h_obs[mb], b_act[mb]
                )

                # Seeker PPO
                s_ratio = (s_lp - b_s_lp_old[mb]).exp()
                s_surr1 = s_ratio * b_s_adv[mb]
                s_surr2 = s_ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_s_adv[mb]
                s_policy_loss = -torch.min(s_surr1, s_surr2).mean()
                s_value_loss  = 0.5 * (s_val_mb - b_s_ret[mb]).pow(2).mean()

                # Hider PPO
                h_ratio = (h_lp - b_h_lp_old[mb]).exp()
                h_surr1 = h_ratio * b_h_adv[mb]
                h_surr2 = h_ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_h_adv[mb]
                h_policy_loss = -torch.min(h_surr1, h_surr2).mean()
                h_value_loss  = 0.5 * (h_val_mb - b_h_ret[mb]).pow(2).mean()

                entropy_loss = -(s_ent + h_ent).mean()

                loss = (
                    s_policy_loss + h_policy_loss
                    + cfg.value_coef   * (s_value_loss + h_value_loss)
                    + cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                s_pol_losses.append(s_policy_loss.item())
                h_pol_losses.append(h_policy_loss.item())
                s_val_losses.append(s_value_loss.item())
                h_val_losses.append(h_value_loss.item())
                entropies.append((s_ent + h_ent).mean().item())

        n = len(s_pol_losses)
        return {
            "s_policy_loss": sum(s_pol_losses) / n,
            "h_policy_loss": sum(h_pol_losses) / n,
            "s_value_loss":  sum(s_val_losses)  / n,
            "h_value_loss":  sum(h_val_losses)  / n,
            "entropy":       sum(entropies)      / n,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int):
        path = Path(self.cfg.checkpoint_dir) / f"policy_{step:09d}.pt"
        torch.save({
            "step":      step,
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)
        print(f"[rl_runner] checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[rl_runner] resumed from {path} (step {ckpt['step']:,})")
        return ckpt["step"]

    # ------------------------------------------------------------------
    # Observation helper
    # ------------------------------------------------------------------

    def _flat_obs(self, obs: Any, role: str) -> torch.Tensor:
        """Flatten the per-agent observation dict into a 1D vector per env."""
        if isinstance(obs, dict):
            agent_obs = obs.get(role, obs)
        else:
            agent_obs = obs

        if isinstance(agent_obs, dict):
            parts = [v.float() for v in agent_obs.values()]
            return torch.cat(parts, dim=-1)
        return agent_obs.float()

    # ------------------------------------------------------------------
    # Video recording helpers
    # ------------------------------------------------------------------

    def _grab_camera_frame(self):
        """Return overhead camera RGB as uint8 HxWx3, or None on failure."""
        import numpy as np
        try:
            self.env.sim.render()
            cam = self.env.scene["overhead_camera"]
            cam.update(dt=self.env.sim.cfg.dt)
            rgb = cam.data.output.get("rgb")
            if rgb is None or rgb.numel() == 0:
                return None
            frame_t = rgb[0, :, :, :3]
            frame = frame_t.cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            return frame if frame.max() > 0 else None
        except Exception:
            return None

    def _write_video(self):
        out_path = Path(self.cfg.video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import cv2
            import numpy as np
            h, w = self._frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.cfg.video_fps,
                (w, h),
            )
            for f in self._frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[rl_runner] video saved → {out_path}")
        except ImportError:
            try:
                import imageio
                imageio.mimwrite(str(out_path), self._frames, fps=self.cfg.video_fps)
                print(f"[rl_runner] video saved → {out_path}")
            except ImportError:
                print("[rl_runner] WARNING: install opencv-python or imageio to save video")
