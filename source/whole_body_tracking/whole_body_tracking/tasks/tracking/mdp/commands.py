from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from whole_body_tracking.tasks.tracking.mdp.motion_info import MotionInfo


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionLibrary:
    """Manages a collection of motions for multi-motion training.

    This class enables loading and efficient access to multiple motion files,
    supporting a single policy network that can execute arbitrary reference motions.

    Attributes:
        motions: Dictionary mapping motion_id to MotionLoader instances
        motion_info: Dictionary mapping motion_id to MotionInfo metadata
        fps_list: List of FPS values for all motions
        device: PyTorch device (cpu or cuda)
    """

    def __init__(self, motions_dir: str, motion_files: list[str], body_indexes: Sequence[int], device: str = "cpu"):
        """Initialize the motion library.

        Args:
            motions_dir: Base directory for motion JSON/CSV metadata
            motion_files: List of NPZ motion file paths
            body_indexes: Indices of bodies to track
            device: PyTorch device
        """
        self.motions: dict[int, MotionLoader] = {}
        self.motion_info: dict[int, MotionInfo] = {}
        self.fps_list: list[int] = []
        self.device = device
        self.num_motions = len(motion_files)

        # Load all motions
        for motion_id, motion_file in enumerate(motion_files):
            motion_loader = MotionLoader(motion_file, body_indexes, device=device)
            self.motions[motion_id] = motion_loader
            self.fps_list.append(motion_loader.fps)

            # Create motion info with default values
            import os
            motion_name = os.path.basename(motion_file).replace(".npz", "")
            info = MotionInfo(
                motion_id=motion_id,
                motion_name=motion_name,
                motion_file=motion_file,
                motion_type="unknown",  # User can override later
                fps=motion_loader.fps,
                num_frames=motion_loader.time_step_total,
                duration_s=motion_loader.time_step_total / motion_loader.fps,
                difficulty_level=0.5,  # Default medium difficulty
            )
            self.motion_info[motion_id] = info

    def get_motion(self, motion_id: int) -> MotionLoader:
        """Get motion by ID."""
        if motion_id not in self.motions:
            raise ValueError(f"Motion ID {motion_id} not found. Available: {list(self.motions.keys())}")
        return self.motions[motion_id]

    def get_motion_info(self, motion_id: int) -> MotionInfo:
        """Get motion metadata by ID."""
        if motion_id not in self.motion_info:
            raise ValueError(f"Motion ID {motion_id} not found in metadata.")
        return self.motion_info[motion_id]

    def get_all_motion_ids(self) -> list[int]:
        """Get list of all available motion IDs."""
        return sorted(self.motions.keys())

    def update_motion_info(self, motion_id: int, info: MotionInfo) -> None:
        """Update metadata for a motion."""
        if motion_id not in self.motion_info:
            raise ValueError(f"Motion ID {motion_id} not found.")
        self.motion_info[motion_id] = info

    def build_stacked_tensors(self) -> dict[str, torch.Tensor]:
        """Build stacked tensors for efficient per-env motion indexing.

        Pre-pads all motions to max_steps with last frame, enabling O(1)
        per-environment motion lookup via: tensor[current_motion_ids, time_steps].

        Returns:
            dict with keys joint_pos, joint_vel, body_pos_w, body_quat_w,
            body_lin_vel_w, body_ang_vel_w (shape [num_motions, max_steps, ...])
            and motion_time_totals (shape [num_motions]).
        """
        time_totals = [self.motions[i].time_step_total for i in range(self.num_motions)]
        max_steps = max(time_totals)
        m0 = self.motions[0]
        num_joints = m0.joint_pos.shape[1]
        num_bodies = m0.body_pos_w.shape[1]
        dev = self.device

        all_joint_pos = torch.zeros(self.num_motions, max_steps, num_joints, device=dev)
        all_joint_vel = torch.zeros(self.num_motions, max_steps, num_joints, device=dev)
        all_body_pos_w = torch.zeros(self.num_motions, max_steps, num_bodies, 3, device=dev)
        all_body_quat_w = torch.zeros(self.num_motions, max_steps, num_bodies, 4, device=dev)
        all_body_lin_vel_w = torch.zeros(self.num_motions, max_steps, num_bodies, 3, device=dev)
        all_body_ang_vel_w = torch.zeros(self.num_motions, max_steps, num_bodies, 3, device=dev)

        for i in range(self.num_motions):
            m = self.motions[i]
            n = m.time_step_total
            all_joint_pos[i, :n] = m.joint_pos
            all_joint_pos[i, n:] = m.joint_pos[-1]
            all_joint_vel[i, :n] = m.joint_vel
            all_joint_vel[i, n:] = m.joint_vel[-1]
            all_body_pos_w[i, :n] = m.body_pos_w
            all_body_pos_w[i, n:] = m.body_pos_w[-1]
            all_body_quat_w[i, :n] = m.body_quat_w
            all_body_quat_w[i, n:] = m.body_quat_w[-1]
            all_body_lin_vel_w[i, :n] = m.body_lin_vel_w
            all_body_lin_vel_w[i, n:] = m.body_lin_vel_w[-1]
            all_body_ang_vel_w[i, :n] = m.body_ang_vel_w
            all_body_ang_vel_w[i, n:] = m.body_ang_vel_w[-1]

        return {
            "joint_pos": all_joint_pos,
            "joint_vel": all_joint_vel,
            "body_pos_w": all_body_pos_w,
            "body_quat_w": all_body_quat_w,
            "body_lin_vel_w": all_body_lin_vel_w,
            "body_ang_vel_w": all_body_ang_vel_w,
            "motion_time_totals": torch.tensor(time_totals, dtype=torch.long, device=dev),
        }

    def __len__(self) -> int:
        """Return number of motions in library."""
        return len(self.motions)

    def __getitem__(self, motion_id: int) -> MotionLoader:
        """Get motion by ID using bracket notation."""
        return self.get_motion(motion_id)


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count


    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


class MultiMotionCommand(CommandTerm):
    """Command term for multi-motion tracking.

    This command term extends MotionCommand to support learning across multiple
    reference motions. It maintains separate adaptive sampling state for each motion
    and tracks the current motion being executed in each environment.
    """

    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # Load motion library
        self.motion_library = MotionLibrary(self.cfg.motion_library_dir, self.cfg.motion_files, self.body_indexes, device=self.device)

        # Update motion info with user-provided metadata
        if self.cfg.motion_info_list is not None:
            for info in self.cfg.motion_info_list:
                self.motion_library.update_motion_info(info.motion_id, info)

        # Build stacked tensors for per-env O(1) motion access
        stacked = self.motion_library.build_stacked_tensors()
        self._all_joint_pos: torch.Tensor = stacked["joint_pos"]
        self._all_joint_vel: torch.Tensor = stacked["joint_vel"]
        self._all_body_pos_w: torch.Tensor = stacked["body_pos_w"]
        self._all_body_quat_w: torch.Tensor = stacked["body_quat_w"]
        self._all_body_lin_vel_w: torch.Tensor = stacked["body_lin_vel_w"]
        self._all_body_ang_vel_w: torch.Tensor = stacked["body_ang_vel_w"]
        self.motion_time_totals: torch.Tensor = stacked["motion_time_totals"]  # [num_motions]

        # Track which motion each environment is currently executing
        self.current_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_change_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Time steps tracking
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Relative coordinates
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Per-motion adaptive sampling state
        self.per_motion_bin_failed_count: dict[int, torch.Tensor] = {}
        self.per_motion_current_bin_failed: dict[int, torch.Tensor] = {}
        self.per_motion_kernel: dict[int, torch.Tensor] = {}

        for motion_id in self.motion_library.get_all_motion_ids():
            motion = self.motion_library.get_motion(motion_id)
            bin_count = int(motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
            self.per_motion_bin_failed_count[motion_id] = torch.zeros(bin_count, dtype=torch.float, device=self.device)
            self.per_motion_current_bin_failed[motion_id] = torch.zeros(bin_count, dtype=torch.float, device=self.device)

            kernel = torch.tensor(
                [self.cfg.adaptive_lambda ** i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
            )
            self.per_motion_kernel[motion_id] = kernel / kernel.sum()

        # Metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["current_motion_id"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_change_count"] = torch.zeros(self.num_envs, device=self.device)

        # Instantiate motion selector (lazy import avoids circular dependency)
        self._init_motion_selector(cfg)

        # Instantiate embedding bank (auto-enable when num_motions >= 50)
        self._init_embedding_bank(cfg)

    # Properties for per-env motion access via stacked tensors
    @property
    def _clamped_time_steps(self) -> torch.Tensor:
        """Time steps clamped to each env's motion duration. Shape: [num_envs]."""
        limits = self.motion_time_totals[self.current_motion_ids] - 1
        return torch.clamp(self.time_steps, max=limits)

    @property
    def command(self) -> torch.Tensor:
        """Get joint position and velocity commands."""
        t = self._clamped_time_steps
        return torch.cat([
            self._all_joint_pos[self.current_motion_ids, t],
            self._all_joint_vel[self.current_motion_ids, t],
        ], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Get reference joint positions."""
        t = self._clamped_time_steps
        return self._all_joint_pos[self.current_motion_ids, t]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Get reference joint velocities."""
        t = self._clamped_time_steps
        return self._all_joint_vel[self.current_motion_ids, t]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Get reference body positions in world frame."""
        t = self._clamped_time_steps
        return self._all_body_pos_w[self.current_motion_ids, t] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Get reference body quaternions in world frame."""
        t = self._clamped_time_steps
        return self._all_body_quat_w[self.current_motion_ids, t]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Get reference body linear velocities in world frame."""
        t = self._clamped_time_steps
        return self._all_body_lin_vel_w[self.current_motion_ids, t]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Get reference body angular velocities in world frame."""
        t = self._clamped_time_steps
        return self._all_body_ang_vel_w[self.current_motion_ids, t]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Get reference anchor position in world frame."""
        t = self._clamped_time_steps
        return self._all_body_pos_w[self.current_motion_ids, t, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Get reference anchor quaternion in world frame."""
        t = self._clamped_time_steps
        return self._all_body_quat_w[self.current_motion_ids, t, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        """Get reference anchor linear velocity in world frame."""
        t = self._clamped_time_steps
        return self._all_body_lin_vel_w[self.current_motion_ids, t, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        """Get reference anchor angular velocity in world frame."""
        t = self._clamped_time_steps
        return self._all_body_ang_vel_w[self.current_motion_ids, t, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def switch_motion(self, env_ids: Sequence[int], motion_ids: torch.Tensor) -> None:
        """Switch motion for specified environments.

        Args:
            env_ids: Indices of environments to update
            motion_ids: New motion IDs for each environment
        """
        self.current_motion_ids[env_ids] = motion_ids
        self.time_steps[env_ids] = 0
        self.motion_change_counts[env_ids] += 1

    def _select_motions(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Select next motion for each resetting environment.

        Uses the attached motion_selector if set, otherwise picks uniformly at random.

        Returns:
            LongTensor of shape [len(env_ids)] with motion IDs.
        """
        if self.motion_selector is not None:
            return self.motion_selector.select_motions(env_ids, self)
        return torch.randint(0, self.motion_library.num_motions, (len(env_ids),), device=self.device)

    def _init_motion_selector(self, cfg: "MultiMotionCommandCfg") -> None:
        """Instantiate the motion selector from config."""
        # Lazy import to avoid circular dependency at module level
        from whole_body_tracking.tasks.tracking.mdp.motion_selector import (
            UniformMotionSelector, AdaptiveMotionSelector, CurriculumMotionSelector,
        )
        selector_type = cfg.motion_selector_type
        if selector_type == "adaptive":
            self.motion_selector = AdaptiveMotionSelector()
        elif selector_type == "curriculum":
            self.motion_selector = CurriculumMotionSelector(
                num_motions=self.motion_library.num_motions,
                success_threshold=cfg.curriculum_success_threshold,
                window=cfg.curriculum_window,
            )
        else:
            self.motion_selector = UniformMotionSelector()

    def _init_embedding_bank(self, cfg: "MultiMotionCommandCfg") -> None:
        """Initialise the MotionEmbeddingBank when embedding mode is enabled.

        Auto-enables when num_motions >= 50 (unless explicitly disabled).
        """
        from whole_body_tracking.utils.motion_embedding import MotionEmbeddingBank
        num_motions = self.motion_library.num_motions
        use_emb = cfg.use_embedding
        if use_emb is None:
            use_emb = (num_motions >= 50)

        if use_emb:
            self.embedding_bank: MotionEmbeddingBank | None = MotionEmbeddingBank(
                num_motions=num_motions,
                embedding_dim=cfg.embedding_dim,
                device=self.device,
            )
            # Initialise from motion metadata if available
            motion_infos = [self.motion_library.get_motion_info(i) for i in range(num_motions)]
            self.embedding_bank.update_from_motion_info(motion_infos)
            print(
                f"[MultiMotionCommand] Embedding bank: {num_motions} motions × "
                f"{cfg.embedding_dim} dims  (use_embedding=True)"
            )
        else:
            self.embedding_bank = None
        """Update error metrics."""
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1)

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)
        self.metrics["current_motion_id"] = self.current_motion_ids.float()
        self.metrics["motion_change_count"] = self.motion_change_counts.float()

    def _adaptive_sampling(self, env_ids: Sequence[int], motion_ids: torch.Tensor) -> None:
        """Perform adaptive sampling within each motion."""
        unique_motions = torch.unique(motion_ids)

        for motion_id in unique_motions:
            motion_id_int = motion_id.item()
            motion_env_ids = env_ids[motion_ids == motion_id]

            if len(motion_env_ids) == 0:
                continue

            motion = self.motion_library.get_motion(motion_id_int)
            bin_count = self.per_motion_bin_failed_count[motion_id_int].shape[0]

            # Update failure tracking
            episode_failed = self._env.termination_manager.terminated[motion_env_ids]
            if torch.any(episode_failed):
                current_bin_index = torch.clamp(
                    (self.time_steps[motion_env_ids] * bin_count) // max(motion.time_step_total, 1), 0, bin_count - 1
                )
                fail_bins = current_bin_index[episode_failed]
                self.per_motion_current_bin_failed[motion_id_int][:] = torch.bincount(
                    fail_bins, minlength=bin_count
                )

            # Sample new time steps
            sampling_probabilities = (
                self.per_motion_bin_failed_count[motion_id_int]
                + self.cfg.adaptive_uniform_ratio / float(bin_count)
            )
            sampling_probabilities = torch.nn.functional.pad(
                sampling_probabilities.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probabilities = torch.nn.functional.conv1d(
                sampling_probabilities, self.per_motion_kernel[motion_id_int].view(1, 1, -1)
            ).view(-1)
            sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

            sampled_bins = torch.multinomial(sampling_probabilities, len(motion_env_ids), replacement=True)
            self.time_steps[motion_env_ids] = (
                (sampled_bins + sample_uniform(0.0, 1.0, (len(motion_env_ids),), device=self.device))
                / bin_count
                * (motion.time_step_total - 1)
            ).long()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Resample command for environments that completed an episode."""
        if len(env_ids) == 0:
            return

        # Notify selector about completed episode outcomes (success vs failure)
        if self.motion_selector is not None:
            episode_failed = self._env.termination_manager.terminated[env_ids]
            success_flags = ~episode_failed  # True = completed without failure
            self.motion_selector.update(
                self.current_motion_ids[env_ids],
                success_flags,
            )

        # Select next motion for each resetting environment
        new_motion_ids = self._select_motions(env_ids)
        self.current_motion_ids[env_ids] = new_motion_ids
        self.motion_change_counts[env_ids] += 1

        # Adaptive sampling within the selected motion
        self._adaptive_sampling(env_ids, new_motion_ids)
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        """Update command for all environments."""
        self.time_steps += 1
        # Per-env time limit: each env has its own motion duration
        current_time_totals = self.motion_time_totals[self.current_motion_ids]  # [num_envs]
        env_ids = torch.where(self.time_steps >= current_time_totals)[0]
        self._resample_command(env_ids)

        # Compute relative coordinates
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # Update per-motion failure tracking
        for motion_id in self.motion_library.get_all_motion_ids():
            self.per_motion_bin_failed_count[motion_id] = (
                self.cfg.adaptive_alpha * self.per_motion_current_bin_failed[motion_id]
                + (1 - self.cfg.adaptive_alpha) * self.per_motion_bin_failed_count[motion_id]
            )
            self.per_motion_current_bin_failed[motion_id].zero_()

        self._update_metrics()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization."""
        # Note: Can be extended in future if needed
        pass

    def _debug_vis_callback(self, event):
        """Debug visualization callback."""
        # Note: Can be extended in future if needed
        pass



@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for multi-motion tracking command.

    Similar to MotionCommandCfg but supports multiple motion files and motion-related observations.
    """

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING
    motion_library_dir: str = MISSING  # Directory for motion metadata (optional)
    motion_files: list[str] = MISSING  # List of NPZ motion file paths
    motion_info_list: list[MotionInfo] | None = None  # Optional pre-configured motion metadata

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    # Motion selector strategy: "uniform", "adaptive", or "curriculum"
    motion_selector_type: str = "uniform"
    # Curriculum: success rate threshold to unlock the next motion
    curriculum_success_threshold: float = 0.7
    # Curriculum: rolling window size for success rate estimation
    curriculum_window: int = 1000

    # Embedding: use continuous embedding instead of one-hot for motion ID.
    # None = auto (True when num_motions >= 50, False otherwise).
    use_embedding: bool | None = None
    # Dimensionality of the motion embedding vector.
    embedding_dim: int = 16