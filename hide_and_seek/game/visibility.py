"""Vision-based detection for hide-and-seek.

Determines whether the seeker can see the hider using:
1. Semantic segmentation from the seeker's camera (primary)
2. Frustum + raycasting check (fallback)
"""

import torch
import math


def check_visibility_semantic(
    semantic_obs: torch.Tensor,
    hider_semantic_id: int,
    pixel_threshold: float = 0.005,
) -> torch.Tensor:
    """Check if hider is visible in seeker's semantic camera output.

    Args:
        semantic_obs: Semantic segmentation tensor from seeker camera,
            shape (num_envs, H, W) or (num_envs, H, W, 1).
        hider_semantic_id: The semantic ID assigned to the hider robot.
        pixel_threshold: Minimum fraction of pixels that must be hider
            for detection (default 0.5%).

    Returns:
        Boolean tensor of shape (num_envs,) — True if hider is visible.
    """
    if semantic_obs.dim() == 4:
        semantic_obs = semantic_obs.squeeze(-1)

    num_envs = semantic_obs.shape[0]
    total_pixels = semantic_obs.shape[1] * semantic_obs.shape[2]

    # Count hider pixels per environment
    hider_pixels = (semantic_obs == hider_semantic_id).view(num_envs, -1).sum(dim=1)
    visibility_ratio = hider_pixels.float() / total_pixels

    return visibility_ratio > pixel_threshold


def check_visibility_frustum(
    seeker_pos: torch.Tensor,
    seeker_forward: torch.Tensor,
    hider_pos: torch.Tensor,
    fov_deg: float = 90.0,
    max_distance: float = 10.0,
) -> torch.Tensor:
    """Geometric frustum check — is hider within seeker's field of view?

    This does NOT check for occlusion. Use with raycasting for full detection.

    Args:
        seeker_pos: Seeker head position, shape (num_envs, 3).
        seeker_forward: Seeker forward direction, shape (num_envs, 3).
        hider_pos: Hider body position, shape (num_envs, 3).
        fov_deg: Field of view in degrees.
        max_distance: Maximum detection range in meters.

    Returns:
        Boolean tensor of shape (num_envs,) — True if hider is in frustum.
    """
    to_hider = hider_pos - seeker_pos
    distance = torch.norm(to_hider, dim=-1)

    # Distance check
    in_range = distance <= max_distance

    # Angle check
    to_hider_normalized = to_hider / (distance.unsqueeze(-1) + 1e-8)
    seeker_forward_normalized = seeker_forward / (
        torch.norm(seeker_forward, dim=-1, keepdim=True) + 1e-8
    )
    cos_angle = (to_hider_normalized * seeker_forward_normalized).sum(dim=-1)
    half_fov_rad = math.radians(fov_deg / 2.0)
    in_fov = cos_angle >= math.cos(half_fov_rad)

    return in_range & in_fov


class VisibilityTracker:
    """Track consecutive visibility detections for confirmed sighting.

    Requires the hider to be visible for N consecutive steps to confirm
    detection, reducing false positives from single-frame glitches.

    Args:
        num_envs: Number of parallel environments.
        confirmation_steps: Number of consecutive visible steps required.
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        confirmation_steps: int = 3,
        device: str = "cuda:0",
    ):
        self.confirmation_steps = confirmation_steps
        self.device = device
        self.consecutive_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, is_visible: torch.Tensor) -> torch.Tensor:
        """Update tracker with new visibility observations.

        Args:
            is_visible: Boolean tensor (num_envs,) of current frame visibility.

        Returns:
            Boolean tensor (num_envs,) — True if detection is confirmed.
        """
        # Increment count where visible, reset where not
        self.consecutive_count = torch.where(
            is_visible,
            self.consecutive_count + 1,
            torch.zeros_like(self.consecutive_count),
        )
        return self.consecutive_count >= self.confirmation_steps

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset tracker for given environments."""
        if env_ids is None:
            self.consecutive_count.zero_()
        else:
            self.consecutive_count[env_ids] = 0
