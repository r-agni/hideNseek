"""Interactive viewer for the hide-and-seek environment.

Control both agents manually with keyboard:
  Seeker: W/A/S/D (forward/left/back/right)
  Hider:  Arrow keys
  R: Reset episode
  Q: Quit

Usage:
    isaaclab -p scripts/interactive_viewer.py
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=False)
    simulation_app = launcher.app

    from hide_and_seek.env.hide_and_seek_env import HideAndSeekEnv
    from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg
    from hide_and_seek.game.phase_manager import GamePhase
    from hide_and_seek.utils.visualization import compose_debug_frame

    import cv2

    print("=" * 60)
    print("Hide-and-Seek Interactive Viewer")
    print("=" * 60)
    print("Controls:")
    print("  Seeker: W/A/S/D")
    print("  Hider:  Arrow keys")
    print("  R: Reset | Q: Quit")
    print("=" * 60)

    cfg = HideAndSeekEnvCfg()
    cfg.scene.num_envs = 1
    env = HideAndSeekEnv(cfg)
    obs, info = env.reset()

    step_count = 0
    running = True

    while running and simulation_app.is_running():
        # Build action from keyboard (polled via OpenCV)
        action = torch.zeros(1, 4, device=env.device)

        # Display camera feeds
        seeker_cam = env.scene["seeker_camera"]
        hider_cam = env.scene["hider_camera"]
        seeker_cam.update(dt=env.step_dt)
        hider_cam.update(dt=env.step_dt)

        seeker_rgb = seeker_cam.data.output["rgb"][0].cpu().numpy().astype(np.uint8)
        hider_rgb = hider_cam.data.output["rgb"][0].cpu().numpy().astype(np.uint8)

        phase = env.phase_manager.phase[0].item()
        timer = env.phase_manager.get_normalized_timer()[0].item()
        detected = env.hider_detected[0].item()

        frame = compose_debug_frame(
            seeker_rgb, hider_rgb, phase, timer, detected, step_count
        )

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Hide and Seek", frame_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            running = False
            break
        elif key == ord("r"):
            obs, info = env.reset()
            step_count = 0
            print("Episode reset!")
            continue

        # Seeker controls (WASD)
        if key == ord("w"):
            action[0, 0] = 1.0  # forward
        elif key == ord("s"):
            action[0, 0] = -1.0  # backward
        elif key == ord("a"):
            action[0, 1] = 1.0  # turn left
        elif key == ord("d"):
            action[0, 1] = -1.0  # turn right

        # Hider controls (arrow keys)
        elif key == 82:  # up arrow
            action[0, 2] = 1.0
        elif key == 84:  # down arrow
            action[0, 2] = -1.0
        elif key == 81:  # left arrow
            action[0, 3] = 1.0
        elif key == 83:  # right arrow
            action[0, 3] = -1.0

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if terminated.any() or truncated.any():
            phase_name = GamePhase(env.phase_manager.phase[0].item()).name
            if detected:
                print(f"Episode ended: HIDER DETECTED at step {step_count}")
            else:
                print(f"Episode ended: {phase_name} at step {step_count}")
            obs, info = env.reset()
            step_count = 0

    cv2.destroyAllWindows()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
