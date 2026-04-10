"""Motion selection strategies for multi-motion RL training.

Provides plug-in selectors that decide which motion to assign to each resetting
environment.  The selector is called once per episode boundary.

Interface
---------
All selectors implement two methods:

    select_motions(env_ids, command) -> LongTensor[len(env_ids)]
        Return the next motion ID for each resetting environment.

    update(motion_ids, success_flags) -> None
        Record episode outcomes.  Called before select_motions so statistics
        are fresh for the selection.

Available selectors
-------------------
- UniformMotionSelector  : uniform random over all motions (default)
- AdaptiveMotionSelector : samples hard (low-success) motions more often
- CurriculumMotionSelector: starts with one motion, unlocks the next when the
                            current one is mastered
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommand


class MotionSelector(ABC):
    """Abstract base class for motion selectors."""

    @abstractmethod
    def select_motions(
        self,
        env_ids: torch.Tensor,
        command: "MultiMotionCommand",
    ) -> torch.Tensor:
        """Return a LongTensor of shape [len(env_ids)] with the next motion ID for each env."""

    @abstractmethod
    def update(
        self,
        motion_ids: torch.Tensor,
        success_flags: torch.Tensor,
    ) -> None:
        """Record completed-episode outcomes.

        Args:
            motion_ids:    LongTensor [N] – the motion that just finished.
            success_flags: BoolTensor [N] – True if the episode succeeded.
        """


# ---------------------------------------------------------------------------
# Uniform (default)
# ---------------------------------------------------------------------------

class UniformMotionSelector(MotionSelector):
    """Sample the next motion uniformly at random over all available motions."""

    def select_motions(
        self,
        env_ids: torch.Tensor,
        command: "MultiMotionCommand",
    ) -> torch.Tensor:
        num_motions = command.motion_library.num_motions
        return torch.randint(0, num_motions, (len(env_ids),), device=command.device)

    def update(self, motion_ids: torch.Tensor, success_flags: torch.Tensor) -> None:
        pass  # No state to update


# ---------------------------------------------------------------------------
# Adaptive (focus on hard motions)
# ---------------------------------------------------------------------------

class AdaptiveMotionSelector(MotionSelector):
    """Sample motions proportional to their difficulty (1 - success_rate).

    This focuses training on motions that are still challenging while still
    occasionally revisiting mastered ones (controlled by ``epsilon``).

    Args:
        alpha:   EMA decay for per-motion success rate (0 < alpha <= 1).
                 Smaller = slower adaptation to recent performance.
        epsilon: Minimum sampling probability for any motion (prevents zero
                 probability for easy motions).
    """

    def __init__(self, alpha: float = 0.01, epsilon: float = 0.05):
        self.alpha = alpha
        self.epsilon = epsilon
        # success_rates is initialised on first use (we don't know num_motions yet)
        self._success_rates: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def _ensure_init(self, num_motions: int, device: torch.device) -> None:
        if self._success_rates is None:
            # Start at 0.5 (neutral) so all motions get equal early exposure
            self._success_rates = torch.full((num_motions,), 0.5, dtype=torch.float, device=device)

    # ------------------------------------------------------------------
    def select_motions(
        self,
        env_ids: torch.Tensor,
        command: "MultiMotionCommand",
    ) -> torch.Tensor:
        num_motions = command.motion_library.num_motions
        self._ensure_init(num_motions, command.device)

        # difficulty = 1 - success_rate (harder motions have higher weight)
        difficulty = 1.0 - self._success_rates.to(command.device)
        probs = difficulty + self.epsilon
        probs = probs / probs.sum()

        return torch.multinomial(probs, len(env_ids), replacement=True)

    # ------------------------------------------------------------------
    def update(self, motion_ids: torch.Tensor, success_flags: torch.Tensor) -> None:
        if self._success_rates is None:
            return

        # EMA update per motion that appeared in this batch
        unique = torch.unique(motion_ids)
        for mid in unique:
            mask = motion_ids == mid
            batch_success_rate = success_flags[mask].float().mean()
            idx = mid.item()
            self._success_rates[idx] = (
                (1 - self.alpha) * self._success_rates[idx] + self.alpha * batch_success_rate
            )


# ---------------------------------------------------------------------------
# Curriculum (stage-based unlock)
# ---------------------------------------------------------------------------

class CurriculumMotionSelector(MotionSelector):
    """Unlock motions one-by-one as the current stage is mastered.

    The first motion (ID 0) is available from the start.  Once the rolling
    success rate of all currently-unlocked motions exceeds
    ``success_threshold``, the next motion is unlocked.  Selection is uniform
    over the currently-unlocked set.

    Args:
        num_motions:       Total number of motions in the library.
        success_threshold: Rolling success rate needed to unlock the next motion.
        window:            Length of the rolling window (in episodes) used to
                           estimate the success rate.
    """

    def __init__(
        self,
        num_motions: int,
        success_threshold: float = 0.7,
        window: int = 1000,
    ):
        self.num_motions = num_motions
        self.success_threshold = success_threshold
        # Per-motion rolling buffers
        self._history: list[deque] = [deque(maxlen=window) for _ in range(num_motions)]
        # How many motions are currently unlocked (starts at 1)
        self._num_unlocked: int = 1

    # ------------------------------------------------------------------
    @property
    def unlocked_motions(self) -> list[int]:
        return list(range(self._num_unlocked))

    # ------------------------------------------------------------------
    def _success_rate(self, motion_id: int) -> float:
        buf = self._history[motion_id]
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    # ------------------------------------------------------------------
    def _maybe_unlock_next(self) -> None:
        """Check whether we should unlock the next motion."""
        if self._num_unlocked >= self.num_motions:
            return
        # All currently-unlocked motions must pass the threshold
        for mid in range(self._num_unlocked):
            if self._success_rate(mid) < self.success_threshold:
                return
        # Unlock next
        self._num_unlocked += 1
        print(
            f"[CurriculumMotionSelector] Unlocked motion {self._num_unlocked - 1}. "
            f"Now training on {self._num_unlocked}/{self.num_motions} motions."
        )

    # ------------------------------------------------------------------
    def select_motions(
        self,
        env_ids: torch.Tensor,
        command: "MultiMotionCommand",
    ) -> torch.Tensor:
        self._maybe_unlock_next()
        # Uniform over unlocked motions
        idx = torch.randint(0, self._num_unlocked, (len(env_ids),), device=command.device)
        return idx

    # ------------------------------------------------------------------
    def update(self, motion_ids: torch.Tensor, success_flags: torch.Tensor) -> None:
        for mid_t, suc_t in zip(motion_ids.tolist(), success_flags.tolist()):
            mid = int(mid_t)
            if 0 <= mid < self.num_motions:
                self._history[mid].append(float(suc_t))
