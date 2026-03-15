"""Visualization utilities for debugging and recording hide-and-seek episodes."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hide_and_seek.env.hide_and_seek_env import HideAndSeekEnv


def compose_debug_frame(
    seeker_rgb: np.ndarray,
    hider_rgb: np.ndarray,
    phase: int,
    timer: float,
    detected: bool,
    step: int,
) -> np.ndarray:
    """Create a side-by-side debug frame with both camera views and status overlay.

    Args:
        seeker_rgb: Seeker camera image, shape (H, W, 3).
        hider_rgb: Hider camera image, shape (H, W, 3).
        phase: Current game phase integer (0-3).
        timer: Normalized phase timer (0-1).
        detected: Whether hider has been detected.
        step: Current simulation step.

    Returns:
        Composite image of shape (H, W*2, 3).
    """
    import cv2

    h, w = seeker_rgb.shape[:2]

    # Side by side
    frame = np.concatenate([seeker_rgb, hider_rgb], axis=1)

    # Phase names
    phase_names = {0: "INIT", 1: "HIDING", 2: "SEEKING", 3: "DONE"}
    phase_name = phase_names.get(phase, "UNKNOWN")

    # Text overlays
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    bg_color = (0, 0, 0)

    # Seeker label
    cv2.putText(frame, "SEEKER", (10, 25), font, 0.7, bg_color, 3)
    cv2.putText(frame, "SEEKER", (10, 25), font, 0.7, color, 1)

    # Hider label
    cv2.putText(frame, "HIDER", (w + 10, 25), font, 0.7, bg_color, 3)
    cv2.putText(frame, "HIDER", (w + 10, 25), font, 0.7, color, 1)

    # Status bar at bottom
    status = f"Phase: {phase_name} | Timer: {timer:.2f} | Step: {step}"
    if detected:
        status += " | DETECTED!"
        color = (0, 0, 255)  # Red for detection
    cv2.putText(frame, status, (10, h - 10), font, 0.5, bg_color, 3)
    cv2.putText(frame, status, (10, h - 10), font, 0.5, color, 1)

    return frame


class EpisodeRecorder:
    """Record episodes to video files.

    Args:
        output_dir: Directory to save video files.
        fps: Frames per second for output video.
    """

    def __init__(self, output_dir: str = "output/videos", fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        self.frames: list[np.ndarray] = []

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the recording."""
        self.frames.append(frame.copy())

    def save(self, filename: str = "episode.mp4"):
        """Save recorded frames to a video file."""
        if not self.frames:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        import imageio

        imageio.mimsave(filepath, self.frames, fps=self.fps)
        print(f"Saved episode recording: {filepath} ({len(self.frames)} frames)")

    def reset(self):
        """Clear recorded frames."""
        self.frames.clear()
