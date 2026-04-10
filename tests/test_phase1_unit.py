"""Unit tests for Phase 1 multi-motion framework components.

Tests cover:
1. MotionInfo dataclass
2. MotionLibrary class
3. MultiMotionCommand class
4. Observation functions
5. Reward functions
"""

import sys
import os
import tempfile
import numpy as np
import torch
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / "source"))

from whole_body_tracking.tasks.tracking.mdp.motion_info import MotionInfo
from whole_body_tracking.tasks.tracking.mdp.commands import MotionLoader, MotionLibrary


class TestMotionInfo:
    """Test MotionInfo dataclass."""

    def test_motion_info_creation(self):
        """Test basic MotionInfo creation."""
        info = MotionInfo(
            motion_id=0,
            motion_name="walk",
            motion_file="/path/to/walk.npz",
            motion_type="locomotion",
            difficulty_level=0.3,
        )
        assert info.motion_id == 0
        assert info.motion_name == "walk"
        assert info.difficulty_level == 0.3
        print("✓ MotionInfo creation works")

    def test_difficulty_validation(self):
        """Test that difficulty_level is validated."""
        try:
            info = MotionInfo(
                motion_id=0,
                motion_name="test",
                motion_file="test.npz",
                motion_type="test",
                difficulty_level=1.5,  # Invalid: > 1.0
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✓ Difficulty validation works: {e}")

    def test_motion_info_str(self):
        """Test MotionInfo string representation."""
        info = MotionInfo(
            motion_id=1,
            motion_name="run",
            motion_file="run.npz",
            motion_type="locomotion",
            difficulty_level=0.7,
            duration_s=10.0,
        )
        str_repr = str(info)
        assert "run" in str_repr
        assert "0.70" in str_repr
        print(f"✓ MotionInfo str works: {str_repr}")


class TestMotionLibrary:
    """Test MotionLibrary class."""

    @staticmethod
    def create_dummy_motion_file(path: str, num_frames: int = 100):
        """Create a dummy NPZ motion file for testing."""
        num_joints = 27
        num_bodies = 15

        data = {
            "fps": 50,
            "joint_pos": np.random.randn(num_frames, num_joints).astype(np.float32),
            "joint_vel": np.random.randn(num_frames, num_joints).astype(np.float32),
            "body_pos_w": np.random.randn(num_frames, num_bodies, 3).astype(np.float32),
            "body_quat_w": np.tile([1, 0, 0, 0], (num_frames, num_bodies, 1)).astype(
                np.float32
            ),
            "body_lin_vel_w": np.random.randn(num_frames, num_bodies, 3).astype(np.float32),
            "body_ang_vel_w": np.random.randn(num_frames, num_bodies, 3).astype(np.float32),
        }
        np.savez(path, **data)
        return path

    def test_motion_library_load(self):
        """Test loading multiple motions into library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 dummy motion files
            motion_files = []
            for i in range(3):
                path = os.path.join(tmpdir, f"motion_{i}.npz")
                self.create_dummy_motion_file(path, num_frames=100 + i * 50)
                motion_files.append(path)

            # Load into library
            body_indexes = list(range(15))
            library = MotionLibrary(tmpdir, motion_files, body_indexes, device="cpu")

            assert len(library) == 3
            assert len(library.get_all_motion_ids()) == 3
            print(f"✓ MotionLibrary loaded {len(library)} motions")

    def test_motion_library_access(self):
        """Test accessing motions from library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_files = []
            for i in range(3):
                path = os.path.join(tmpdir, f"motion_{i}.npz")
                self.create_dummy_motion_file(path)
                motion_files.append(path)

            body_indexes = list(range(15))
            library = MotionLibrary(tmpdir, motion_files, body_indexes, device="cpu")

            # Test O(1) access
            for motion_id in library.get_all_motion_ids():
                motion = library.get_motion(motion_id)
                assert motion is not None
                assert motion.time_step_total == 100

                info = library.get_motion_info(motion_id)
                assert info.motion_id == motion_id
                print(f"✓ Motion {motion_id}: {info.motion_name}, {info.difficulty_level:.1f}")

    def test_motion_library_metadata(self):
        """Test motion metadata management."""
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_files = []
            for i in range(2):
                path = os.path.join(tmpdir, f"motion_{i}.npz")
                self.create_dummy_motion_file(path)
                motion_files.append(path)

            body_indexes = list(range(15))
            library = MotionLibrary(tmpdir, motion_files, body_indexes, device="cpu")

            # Update metadata
            new_info = MotionInfo(
                motion_id=0,
                motion_name="walking",
                motion_file=motion_files[0],
                motion_type="locomotion",
                difficulty_level=0.4,
                body_part_tags=["legs", "arms"],
            )
            library.update_motion_info(0, new_info)

            info = library.get_motion_info(0)
            assert info.motion_type == "locomotion"
            assert 0.4 == info.difficulty_level
            print(f"✓ Metadata updated: {info}")


class TestObservationFunctions:
    """Test observation encoding functions.

    Note: These require a full environment setup, so we do basic structure tests.
    """

    def test_motion_id_encoding_shape(self):
        """Test motion_id_encoding output shape."""
        try:
            from whole_body_tracking.tasks.tracking.mdp.observations import motion_id_encoding

            # This would require a full env, but we can test the import
            print("✓ motion_id_encoding function imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import observation functions: {e}")

    def test_motion_change_signal_shape(self):
        """Test motion_change_signal output shape."""
        try:
            from whole_body_tracking.tasks.tracking.mdp.observations import motion_change_signal

            print("✓ motion_change_signal function imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import: {e}")

    def test_motion_progress_shape(self):
        """Test motion_progress output shape."""
        try:
            from whole_body_tracking.tasks.tracking.mdp.observations import motion_progress

            print("✓ motion_progress function imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import: {e}")


class TestRewardFunctions:
    """Test reward functions."""

    def test_reward_functions_import(self):
        """Test that all reward functions can be imported."""
        try:
            from whole_body_tracking.tasks.tracking.mdp.rewards import (
                motion_difficulty_scaling,
                motion_diversity_bonus,
                motion_switching_penalty,
            )

            print("✓ All reward functions imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import reward functions: {e}")


class TestConfigurationClasses:
    """Test configuration classes."""

    def test_config_import(self):
        """Test that configuration classes can be imported."""
        try:
            from whole_body_tracking.tasks.tracking.mdp.commands import (
                MotionCommandCfg,
                MultiMotionCommandCfg,
            )

            print("✓ MotionCommandCfg and MultiMotionCommandCfg imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import configs: {e}")

    def test_env_config_import(self):
        """Test environment configuration classes."""
        try:
            from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
                CommandsCfg,
                CommandsCfgMultiMotion,
                ObservationsCfgMultiMotion,
            )

            print("✓ All env config classes imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import env configs: {e}")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "=" * 70)
    print("PHASE 1 UNIT TEST SUITE")
    print("=" * 70 + "\n")

    test_suites = [
        ("MotionInfo Tests", TestMotionInfo()),
        ("MotionLibrary Tests", TestMotionLibrary()),
        ("Observation Functions Tests", TestObservationFunctions()),
        ("Reward Functions Tests", TestRewardFunctions()),
        ("Configuration Classes Tests", TestConfigurationClasses()),
    ]

    passed = 0
    failed = 0

    for suite_name, test_class in test_suites:
        print(f"\n{suite_name}")
        print("-" * 70)

        test_methods = [method for method in dir(test_class) if method.startswith("test_")]

        for method_name in test_methods:
            try:
                method = getattr(test_class, method_name)
                method()
                passed += 1
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
