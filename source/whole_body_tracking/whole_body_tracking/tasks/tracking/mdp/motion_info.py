"""Motion metadata and information structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MotionInfo:
    """Metadata for a single motion in the library.

    Attributes:
        motion_id: Unique identifier for this motion (0-based index)
        motion_name: Human-readable name of the motion
        motion_file: Path to the NPZ motion data file
        motion_type: Semantic type of motion (e.g., "walk", "run", "dance", "jump")
        difficulty_level: Difficulty score from 0.0 (easiest) to 1.0 (hardest), used for curriculum
        fps: Frames per second of the motion data
        duration_s: Total duration of the motion in seconds
        num_frames: Total number of frames in the motion
        body_part_tags: Optional tags describing which body parts are active (e.g., "arms", "legs", "full_body")
        description: Optional human-readable description
    """

    motion_id: int
    motion_name: str
    motion_file: str
    motion_type: str
    fps: int = 30
    duration_s: float = 0.0
    num_frames: int = 0
    difficulty_level: float = 0.5  # 0.0 = easy, 1.0 = hard
    body_part_tags: list[str] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        """Validate motion info."""
        if not 0.0 <= self.difficulty_level <= 1.0:
            raise ValueError(f"difficulty_level must be in [0.0, 1.0], got {self.difficulty_level}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.duration_s < 0:
            raise ValueError(f"duration_s must be non-negative, got {self.duration_s}")
        if self.num_frames < 0:
            raise ValueError(f"num_frames must be non-negative, got {self.num_frames}")

    def __str__(self) -> str:
        """String representation."""
        return (
            f"MotionInfo(id={self.motion_id}, name={self.motion_name}, "
            f"type={self.motion_type}, difficulty={self.difficulty_level:.2f}, "
            f"duration={self.duration_s:.2f}s)"
        )

    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()
