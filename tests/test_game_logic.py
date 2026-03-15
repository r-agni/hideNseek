"""Unit tests for game phase manager and visibility detection.

These tests don't require Isaac Lab / GPU — they test pure torch logic.
"""

import torch
import pytest

from hide_and_seek.game.phase_manager import GamePhase, PhaseManager
from hide_and_seek.game.visibility import (
    VisibilityTracker,
    check_visibility_frustum,
)


class TestPhaseManager:
    """Tests for the game phase state machine."""

    def setup_method(self):
        self.pm = PhaseManager(
            num_envs=4, hiding_steps=10, seeking_steps=20, device="cpu"
        )

    def test_initial_state(self):
        assert (self.pm.phase == GamePhase.INIT).all()
        assert (self.pm.phase_step == 0).all()

    def test_init_to_hiding(self):
        self.pm.step()
        assert (self.pm.phase == GamePhase.HIDING).all()

    def test_hiding_to_seeking(self):
        # Step through INIT + full HIDING phase
        for _ in range(1 + 10):
            self.pm.step()
        assert (self.pm.phase == GamePhase.SEEKING).all()

    def test_seeking_to_done(self):
        # Step through all phases
        for _ in range(1 + 10 + 20):
            self.pm.step()
        assert (self.pm.phase == GamePhase.DONE).all()

    def test_reset_specific_envs(self):
        # Advance all envs to seeking
        for _ in range(15):
            self.pm.step()
        assert (self.pm.phase == GamePhase.SEEKING).all()

        # Reset only envs 0 and 2
        self.pm.reset(torch.tensor([0, 2]))
        assert self.pm.phase[0] == GamePhase.INIT
        assert self.pm.phase[1] == GamePhase.SEEKING
        assert self.pm.phase[2] == GamePhase.INIT
        assert self.pm.phase[3] == GamePhase.SEEKING

    def test_normalized_timer(self):
        self.pm.step()  # INIT → HIDING
        timer = self.pm.get_normalized_timer()
        assert (timer >= 0).all() and (timer <= 1).all()

    def test_phase_masks(self):
        self.pm.step()  # → HIDING
        assert self.pm.is_hiding().all()
        assert not self.pm.is_seeking().any()
        assert not self.pm.is_done().any()


class TestVisibilityFrustum:
    """Tests for geometric frustum visibility check."""

    def test_in_fov(self):
        seeker_pos = torch.tensor([[0.0, 0.0, 0.0]])
        seeker_forward = torch.tensor([[1.0, 0.0, 0.0]])
        hider_pos = torch.tensor([[5.0, 0.0, 0.0]])  # Directly ahead

        result = check_visibility_frustum(
            seeker_pos, seeker_forward, hider_pos, fov_deg=90.0, max_distance=10.0
        )
        assert result[0].item() is True

    def test_out_of_fov(self):
        seeker_pos = torch.tensor([[0.0, 0.0, 0.0]])
        seeker_forward = torch.tensor([[1.0, 0.0, 0.0]])
        hider_pos = torch.tensor([[-5.0, 0.0, 0.0]])  # Behind seeker

        result = check_visibility_frustum(
            seeker_pos, seeker_forward, hider_pos, fov_deg=90.0, max_distance=10.0
        )
        assert result[0].item() is False

    def test_out_of_range(self):
        seeker_pos = torch.tensor([[0.0, 0.0, 0.0]])
        seeker_forward = torch.tensor([[1.0, 0.0, 0.0]])
        hider_pos = torch.tensor([[20.0, 0.0, 0.0]])  # Too far

        result = check_visibility_frustum(
            seeker_pos, seeker_forward, hider_pos, fov_deg=90.0, max_distance=10.0
        )
        assert result[0].item() is False

    def test_edge_of_fov(self):
        seeker_pos = torch.tensor([[0.0, 0.0, 0.0]])
        seeker_forward = torch.tensor([[1.0, 0.0, 0.0]])
        # Just inside 90° FOV (45° half-angle)
        hider_pos = torch.tensor([[5.0, 4.9, 0.0]])

        result = check_visibility_frustum(
            seeker_pos, seeker_forward, hider_pos, fov_deg=90.0, max_distance=20.0
        )
        assert result[0].item() is True

    def test_batched(self):
        seeker_pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        seeker_forward = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        hider_pos = torch.tensor([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])

        result = check_visibility_frustum(
            seeker_pos, seeker_forward, hider_pos, fov_deg=90.0, max_distance=10.0
        )
        assert result[0].item() is True
        assert result[1].item() is False


class TestVisibilityTracker:
    """Tests for consecutive detection confirmation."""

    def test_single_frame_not_confirmed(self):
        tracker = VisibilityTracker(num_envs=1, confirmation_steps=3, device="cpu")
        visible = torch.tensor([True])
        confirmed = tracker.update(visible)
        assert confirmed[0].item() is False

    def test_confirmed_after_n_frames(self):
        tracker = VisibilityTracker(num_envs=1, confirmation_steps=3, device="cpu")
        for i in range(3):
            confirmed = tracker.update(torch.tensor([True]))
        assert confirmed[0].item() is True

    def test_reset_on_loss(self):
        tracker = VisibilityTracker(num_envs=1, confirmation_steps=3, device="cpu")
        tracker.update(torch.tensor([True]))
        tracker.update(torch.tensor([True]))
        tracker.update(torch.tensor([False]))  # Lost sight
        confirmed = tracker.update(torch.tensor([True]))
        assert confirmed[0].item() is False  # Counter reset

    def test_reset_envs(self):
        tracker = VisibilityTracker(num_envs=2, confirmation_steps=2, device="cpu")
        tracker.update(torch.tensor([True, True]))
        tracker.reset(torch.tensor([0]))
        confirmed = tracker.update(torch.tensor([True, True]))
        assert confirmed[0].item() is False  # Env 0 was reset
        assert confirmed[1].item() is True  # Env 1 reached 2 consecutive


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
