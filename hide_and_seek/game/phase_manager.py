"""Game phase state machine for hide-and-seek episodes."""

from enum import IntEnum

import torch


class GamePhase(IntEnum):
    """Phases of a hide-and-seek episode."""

    INIT = 0  # Agents placed, transitions immediately
    HIDING = 1  # Hider moves, seeker frozen
    SEEKING = 2  # Both move, seeker hunts
    DONE = 3  # Episode over


class PhaseManager:
    """Manages phase transitions for batched environments.

    Tracks the current phase and step count per environment,
    advancing through INIT → HIDING → SEEKING → DONE.

    Args:
        num_envs: Number of parallel environments.
        hiding_steps: Duration of hiding phase in simulation steps.
        seeking_steps: Duration of seeking phase in simulation steps.
        device: Torch device for tensors.
    """

    def __init__(
        self,
        num_envs: int,
        hiding_steps: int = 150,
        seeking_steps: int = 900,
        device: str = "cuda:0",
    ):
        self.num_envs = num_envs
        self.hiding_steps = hiding_steps
        self.seeking_steps = seeking_steps
        self.device = device

        # Per-environment state
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_step = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset phase state for given environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self.phase[env_ids] = GamePhase.INIT
        self.phase_step[env_ids] = 0

    def step(self) -> torch.Tensor:
        """Advance all environments by one step, handling phase transitions.

        Returns:
            Current phase for each environment.
        """
        self.phase_step += 1

        # INIT → HIDING (immediate)
        init_mask = self.phase == GamePhase.INIT
        if init_mask.any():
            self.phase[init_mask] = GamePhase.HIDING
            self.phase_step[init_mask] = 0

        # HIDING → SEEKING
        hiding_done = (self.phase == GamePhase.HIDING) & (
            self.phase_step >= self.hiding_steps
        )
        if hiding_done.any():
            self.phase[hiding_done] = GamePhase.SEEKING
            self.phase_step[hiding_done] = 0

        # SEEKING → DONE
        seeking_done = (self.phase == GamePhase.SEEKING) & (
            self.phase_step >= self.seeking_steps
        )
        if seeking_done.any():
            self.phase[seeking_done] = GamePhase.DONE

        return self.phase

    def get_normalized_timer(self) -> torch.Tensor:
        """Get normalized time remaining in current phase (1.0 = full, 0.0 = done).

        Returns:
            Tensor of shape (num_envs,) with values in [0, 1].
        """
        timer = torch.zeros(self.num_envs, device=self.device)

        hiding_mask = self.phase == GamePhase.HIDING
        if hiding_mask.any():
            timer[hiding_mask] = 1.0 - self.phase_step[hiding_mask].float() / self.hiding_steps

        seeking_mask = self.phase == GamePhase.SEEKING
        if seeking_mask.any():
            timer[seeking_mask] = 1.0 - self.phase_step[seeking_mask].float() / self.seeking_steps

        return timer.clamp(0.0, 1.0)

    def is_hiding(self) -> torch.Tensor:
        """Returns boolean mask of environments in hiding phase."""
        return self.phase == GamePhase.HIDING

    def is_seeking(self) -> torch.Tensor:
        """Returns boolean mask of environments in seeking phase."""
        return self.phase == GamePhase.SEEKING

    def is_done(self) -> torch.Tensor:
        """Returns boolean mask of environments that are done."""
        return self.phase == GamePhase.DONE
