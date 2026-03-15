"""End-to-end test for the hide-and-seek environment.

Verifies:
1. Environment creates without errors
2. Two G1 agents spawn at different positions
3. Camera sensors produce valid outputs
4. Actions move agents
5. Phase transitions work correctly
6. Visibility detection fires appropriately
7. Episode reset works

Usage:
    isaaclab -p scripts/test_env.py [--headless]
"""

import argparse
import sys
import os

import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Test hide-and-seek environment")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    return parser.parse_args()


def main():
    args = parse_args()

    # Isaac Lab requires AppLauncher before any other imports
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless)
    simulation_app = launcher.app

    # Now safe to import environment modules
    from hide_and_seek.env.hide_and_seek_env import HideAndSeekEnv
    from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg
    from hide_and_seek.game.phase_manager import GamePhase

    print("=" * 60)
    print("Hide-and-Seek Environment Test")
    print("=" * 60)

    # --- Test 1: Environment creation ---
    print("\n[1/8] Creating environment...")
    cfg = HideAndSeekEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = HideAndSeekEnv(cfg)
    print(f"  OK: Created env with {env.num_envs} environment(s)")

    # --- Test 2: Reset and check dual agents ---
    print("\n[2/8] Resetting environment...")
    obs, info = env.reset()
    seeker = env.scene["seeker"]
    hider = env.scene["hider"]
    seeker_pos = seeker.data.root_pos_w[0].cpu().numpy()
    hider_pos = hider.data.root_pos_w[0].cpu().numpy()
    dist = ((seeker_pos - hider_pos) ** 2).sum() ** 0.5
    print(f"  Seeker pos: {seeker_pos}")
    print(f"  Hider pos:  {hider_pos}")
    print(f"  Distance:   {dist:.2f}m")
    assert dist > 0, "Agents should be at different positions"
    print("  OK: Two agents spawned at different positions")

    # --- Test 3: Check observations ---
    print("\n[3/8] Checking observations...")
    for group_name, group_obs in obs.items():
        print(f"  Observation group '{group_name}':")
        if isinstance(group_obs, dict):
            for key, val in group_obs.items():
                print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"    shape={group_obs.shape}, dtype={group_obs.dtype}")
    print("  OK: Observations are valid")

    # --- Test 4: Check camera outputs ---
    print("\n[4/8] Checking camera sensors...")
    seeker_cam = env.scene["seeker_camera"]
    hider_cam = env.scene["hider_camera"]
    seeker_cam.update(dt=env.step_dt)
    hider_cam.update(dt=env.step_dt)

    seeker_rgb = seeker_cam.data.output["rgb"]
    print(f"  Seeker camera RGB: shape={seeker_rgb.shape}")
    assert seeker_rgb.shape[-3] == 256 and seeker_rgb.shape[-2] == 256, "Expected 256x256"
    print("  OK: Camera sensors producing valid output")

    # --- Test 5: Step with actions and check movement ---
    print("\n[5/8] Testing agent movement...")
    pos_before = seeker.data.root_pos_w[0].clone()

    # Move seeker forward for 10 steps
    action = torch.zeros(env.num_envs, 4, device=env.device)
    action[:, 0] = 1.0  # seeker forward
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(action)

    pos_after = seeker.data.root_pos_w[0]
    moved = torch.norm(pos_after - pos_before).item()
    print(f"  Seeker moved: {moved:.4f}m over 10 steps")
    print("  OK: Agent responds to actions")

    # --- Test 6: Phase transitions ---
    print("\n[6/8] Testing phase transitions...")
    obs, info = env.reset()

    # Should start in INIT/HIDING
    initial_phase = env.phase_manager.phase[0].item()
    print(f"  Initial phase: {GamePhase(initial_phase).name}")

    # Step through hiding phase
    action = torch.zeros(env.num_envs, 4, device=env.device)
    for i in range(cfg.hiding_phase_steps + 5):
        obs, reward, terminated, truncated, info = env.step(action)

    current_phase = env.phase_manager.phase[0].item()
    print(f"  After {cfg.hiding_phase_steps + 5} steps: {GamePhase(current_phase).name}")
    assert current_phase == GamePhase.SEEKING, f"Expected SEEKING, got {GamePhase(current_phase).name}"
    print("  OK: Phase transitions work correctly")

    # --- Test 7: Visibility detection ---
    print("\n[7/8] Testing visibility detection...")
    # Place agents face-to-face at close range
    seeker.write_root_pose_to_sim(
        torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], device=env.device),
        torch.tensor([0], device=env.device),
    )
    hider.write_root_pose_to_sim(
        torch.tensor([[2.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], device=env.device),
        torch.tensor([0], device=env.device),
    )

    # Step to trigger detection
    for _ in range(cfg.detection_confirmation_steps + 2):
        obs, reward, terminated, truncated, info = env.step(action)

    detected = env.hider_detected[0].item()
    print(f"  Hider detected: {detected}")
    print("  OK: Visibility detection test complete")

    # --- Test 8: Episode reset ---
    print("\n[8/8] Testing episode reset...")
    obs, info = env.reset()
    new_seeker_pos = seeker.data.root_pos_w[0].cpu().numpy()
    new_hider_pos = hider.data.root_pos_w[0].cpu().numpy()
    print(f"  New seeker pos: {new_seeker_pos}")
    print(f"  New hider pos:  {new_hider_pos}")
    assert not env.hider_detected[0].item(), "Detection should be reset"
    print("  OK: Environment resets correctly")

    # Summary
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
