"""Test: send forward velocity command and check if robot walks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False)

import torch
from hide_and_seek.env.hide_and_seek_env import HideAndSeekEnv
from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg
from pathlib import Path

policy_path = ""
for p in [Path.home() / "unitree_rl_gym/deploy/pre_train/g1/motion.pt",
          Path(__file__).resolve().parents[1] / "data/models/g1_motion.pt"]:
    if p.exists():
        policy_path = str(p)
        break

cfg = HideAndSeekEnvCfg()
cfg.scene.num_envs = 1
cfg.locomotion_policy_path = policy_path

env = HideAndSeekEnv(cfg)
obs, _ = env.reset()

seeker = env.scene["seeker"]

# Send modest forward walk command: seeker vx=0.3 (gentle), hider stays still
# Actions: [seeker_vx, seeker_vy, seeker_yaw, hider_vx, hider_vy, hider_yaw]
walk_action = torch.tensor([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]], device=env.device)

start_x = seeker.data.root_pos_w[0, 0].item()
start_y = seeker.data.root_pos_w[0, 1].item()

print(f"\n=== WALK TEST: seeker vx=1.0 (forward) ===")
print(f"Start pos: x={start_x:.3f}, y={start_y:.3f}, z={seeker.data.root_pos_w[0,2].item():.3f}")

for step in range(200):
    obs, rew, terminated, truncated, info = env.step(walk_action)

    if terminated[0].item():
        print(f"Step {step}: TERMINATED (fallen) — resetting")
        obs, _ = env.reset()
        start_x = seeker.data.root_pos_w[0, 0].item()
        start_y = seeker.data.root_pos_w[0, 1].item()
        continue

    if step % 25 == 0:
        x = seeker.data.root_pos_w[0, 0].item()
        y = seeker.data.root_pos_w[0, 1].item()
        z = seeker.data.root_pos_w[0, 2].item()
        dx = x - start_x
        dy = y - start_y
        dist = (dx**2 + dy**2)**0.5
        print(f"Step {step:3d}: x={x:.3f} y={y:.3f} z={z:.3f}  moved={dist:.3f}m  "
              f"phase={env.phase_manager.phase[0].item():.0f}")

x = seeker.data.root_pos_w[0, 0].item()
y = seeker.data.root_pos_w[0, 1].item()
z = seeker.data.root_pos_w[0, 2].item()
total_dist = ((x - start_x)**2 + (y - start_y)**2)**0.5
print(f"\n=== RESULT ===")
print(f"Final: x={x:.3f} y={y:.3f} z={z:.3f}")
print(f"Total distance moved: {total_dist:.3f}m")
print(f"Standing: {'YES' if z > 0.4 else 'NO (fallen)'}")
print(f"Walking: {'YES' if total_dist > 0.5 else 'NO (stayed in place)'}")

env.close()
launcher.app.close()
