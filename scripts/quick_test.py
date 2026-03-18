"""Quick test: load env, step, print robot heights + locomotion debug."""
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

seeker = env.scene["seeker"]
hider = env.scene["hider"]

# Check joint positions BEFORE first reset
print(f"\n=== BEFORE RESET ===")
print(f"Seeker joint_pos[hip_pitch(0)]: {seeker.data.joint_pos[0, 0].item():.4f}")
print(f"Seeker joint_pos[knee(11)]: {seeker.data.joint_pos[0, 11].item():.4f}")

obs, _ = env.reset()

# Check AFTER reset
print(f"\n=== AFTER RESET ===")
print(f"Seeker joint_pos[hip_pitch(0)]: {seeker.data.joint_pos[0, 0].item():.4f}")
print(f"Seeker joint_pos[knee(11)]: {seeker.data.joint_pos[0, 11].item():.4f}")

loco = env._seeker_loco

print(f"\n=== LOCOMOTION POLICY DEBUG ===")
print(f"Policy loaded: {loco.loaded}")
print(f"Leg joint indices: {loco._leg_joint_indices}")
if loco._leg_joint_indices is not None:
    idx = loco._leg_joint_indices
    names = seeker.data.joint_names
    print(f"Mapped joints: {[names[i] for i in idx.tolist()]}")
    print(f"Current leg positions: {seeker.data.joint_pos[0, idx].cpu().tolist()}")
print(f"Default joint pos: {loco.default_joint_pos.cpu().tolist()}")
print(f"Gravity: {seeker.data.projected_gravity_b[0].cpu().tolist()}")

# Test locomotion policy directly with zero command
cmd = torch.zeros(1, 3, device=env.device)
targets = loco.step(seeker, cmd)
print(f"\nZero cmd → joint targets: {targets[0].cpu().tolist()}")
print(f"Targets - defaults = {(targets[0] - loco.default_joint_pos).cpu().tolist()}")

# Now run 60 steps
print(f"\n=== STEPPING ===")
for step in range(60):
    actions = torch.zeros(1, 6, device=env.device)
    obs, rew, terminated, truncated, info = env.step(actions)

    s_z = seeker.data.root_pos_w[0, 2].item()
    if step % 10 == 0:
        grav = seeker.data.projected_gravity_b[0].cpu().tolist()
        print(f"Step {step:3d}: z={s_z:.3f}  gravity_b={[f'{g:.2f}' for g in grav]}  "
              f"terminated={terminated[0].item()}")

s_z = seeker.data.root_pos_w[0, 2].item()
print(f"\nFinal seeker z = {s_z:.3f} (standing ~0.74, fallen <0.3)")

env.close()
launcher.app.close()
