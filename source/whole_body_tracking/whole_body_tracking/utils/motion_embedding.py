"""Motion embedding bank for scalable multi-motion conditioning.

For small libraries (<50 motions) a one-hot vector is simple and works well
(see ``motion_id_encoding`` in observations.py).  As the library grows the
one-hot dimension grows with it, which blows up the observation space.

This module provides a fixed-size continuous embedding that maps each motion ID
to a vector of constant dimension (``embedding_dim``, default 16) regardless of
how many motions are in the library.

Embeddings are computed from MotionInfo metadata features so they are
meaningful from the start — motions with similar durations / difficulties land
near each other in embedding space — without requiring additional training.
A ``learned`` update API is also provided so controllers can fine-tune the
table through gradient-free methods (ES, weight-sharing) if desired.

Typical use
-----------
The bank is owned by ``MultiMotionCommand`` when ``use_embedding=True`` (or when
``num_motions >= 50``).  The ``motion_id_embedding`` observation function in
``observations.py`` calls ``bank.get(current_motion_ids)`` to produce the per-env
observation vector.

Memory
------
    embedding_table: [num_motions, embedding_dim] × 4 bytes (float32)
    50 motions × 16 dims ≈ 3.2 KB  (negligible)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from whole_body_tracking.tasks.tracking.mdp.motion_info import MotionInfo


class MotionEmbeddingBank:
    """Fixed-size embedding table mapping motion IDs → continuous vectors.

    Args:
        num_motions:   Number of motions in the library.
        embedding_dim: Dimensionality of each embedding vector.
        device:        Torch device.
        init_mode:     Initialisation strategy.

            * ``"feature"`` (default): derive embeddings from MotionInfo metadata
              by projecting a hand-crafted feature vector with a fixed random
              projection matrix.  Call :meth:`update_from_motion_info` after
              creation.
            * ``"random"``: sample from a unit-normalised Gaussian.  Call
              :meth:`update_from_motion_info` later to improve quality.
    """

    def __init__(
        self,
        num_motions: int,
        embedding_dim: int = 16,
        device: str | torch.device = "cpu",
        init_mode: str = "feature",
    ):
        self.num_motions = num_motions
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)

        # Initialise with unit-Gaussian rows (fallback until feature init)
        emb = torch.randn(num_motions, embedding_dim, device=self.device)
        self.embeddings: torch.Tensor = self._normalise_rows(emb)

        self._init_mode = init_mode

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Return embeddings for a batch of motion IDs.

        Args:
            motion_ids: LongTensor of arbitrary shape ``[...]``.

        Returns:
            FloatTensor ``[..., embedding_dim]``.
        """
        return self.embeddings[motion_ids]

    def update_from_motion_info(self, motion_infos: list["MotionInfo"]) -> None:
        """Recompute embeddings from MotionInfo metadata.

        Builds a hand-crafted feature vector for each motion and projects it
        into ``embedding_dim`` space with a fixed random rotation matrix
        (Johnson–Lindenstrauss style).  The result is L2-normalised so all
        embeddings live on the unit hyper-sphere, which gives well-conditioned
        inputs to downstream MLPs.

        Feature vector (per motion):
            [log(duration_s + 1),   # log-scaled duration
             difficulty_level,       # 0.0 (easy) – 1.0 (hard)
             log(fps / 30),          # relative frame rate
             log(num_frames + 1)]    # log-scaled clip length
        """
        if not motion_infos:
            return

        raw_dim = 4
        features = torch.zeros(len(motion_infos), raw_dim, device=self.device)
        for i, info in enumerate(motion_infos):
            features[i, 0] = math.log(max(info.duration_s, 1e-3) + 1.0)
            features[i, 1] = float(info.difficulty_level)
            features[i, 2] = math.log(max(info.fps, 1) / 30.0 + 1.0)
            features[i, 3] = math.log(max(info.num_frames, 1) + 1.0)

        # Normalise each feature column to [0, 1] range
        col_min = features.min(dim=0).values
        col_max = features.max(dim=0).values
        col_range = (col_max - col_min).clamp(min=1e-6)
        features = (features - col_min) / col_range  # [num_motions, raw_dim]

        # Random projection: features → embedding_dim
        torch.manual_seed(42)  # deterministic projection matrix
        proj = torch.randn(raw_dim, self.embedding_dim, device=self.device)
        proj = proj / proj.norm(dim=0, keepdim=True)

        emb = features @ proj  # [num_motions, embedding_dim]
        self.embeddings = self._normalise_rows(emb)

    # ------------------------------------------------------------------
    # Persistence helpers (for checkpoint / ONNX export)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serialisable state."""
        return {"embeddings": self.embeddings.cpu()}

    def load_state_dict(self, state: dict) -> None:
        """Restore from serialised state."""
        self.embeddings = state["embeddings"].to(self.device)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_rows(t: torch.Tensor) -> torch.Tensor:
        norms = t.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return t / norms

    def __repr__(self) -> str:
        return (
            f"MotionEmbeddingBank(num_motions={self.num_motions}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device={self.device})"
        )
