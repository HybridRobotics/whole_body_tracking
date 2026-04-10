# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from whole_body_tracking.tasks.tracking.mdp import MotionCommand


# ---------------------------------------------------------------------------
# Single-motion exporter (original)
# ---------------------------------------------------------------------------

def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    """Export a single-motion policy to ONNX.

    Auto-detects whether the ``"motion"`` command is a
    :class:`MultiMotionCommand` and delegates to
    :func:`export_multi_motion_policy_as_onnx` when appropriate.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # Auto-detect multi-motion
    from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommand
    cmd = env.command_manager.get_term("motion")
    if isinstance(cmd, MultiMotionCommand):
        export_multi_motion_policy_as_onnx(env, actor_critic, path, normalizer, filename, verbose)
        return

    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


# ---------------------------------------------------------------------------
# Multi-motion exporter (Phase 3)
# ---------------------------------------------------------------------------

def export_multi_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
) -> None:
    """Export a multi-motion policy to ONNX.

    The ONNX model takes three inputs:

    * ``obs``       – flat observation vector  ``[1, obs_dim]``
    * ``motion_id`` – integer motion index     ``[1, 1]``
    * ``time_step`` – integer frame number     ``[1, 1]``

    And returns actions plus all motion reference data for the requested
    motion at the requested frame (same 7-output contract as the single-motion
    exporter).
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    exporter = _OnnxMultiMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    exporter.export(path, filename)


class _OnnxMultiMotionPolicyExporter(_OnnxPolicyExporter):
    """ONNX exporter for :class:`MultiMotionCommand`-based policies.

    All motion reference data is baked into the ONNX graph as constants
    using a flat ``[num_motions * max_steps, ...]`` layout for ONNX-safe
    1D integer indexing:

        flat_idx = motion_id * max_steps + time_step
        joint_pos_out = all_joint_pos_flat[flat_idx]
    """

    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)

        from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommand
        cmd: MultiMotionCommand = env.command_manager.get_term("motion")

        num_motions = cmd.motion_library.num_motions
        max_steps = int(cmd._all_joint_pos.shape[1])
        n_bodies = int(cmd._all_body_pos_w.shape[2])

        # Flatten [num_motions, max_steps, ...] → [num_motions * max_steps, ...]
        # for ONNX-safe 1-D indexed lookup.
        def _flat(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(num_motions * max_steps, *t.shape[2:]).to("cpu")

        self.all_joint_pos     = _flat(cmd._all_joint_pos)       # [NM, J]
        self.all_joint_vel     = _flat(cmd._all_joint_vel)       # [NM, J]
        self.all_body_pos_w    = _flat(cmd._all_body_pos_w)      # [NM, B, 3]
        self.all_body_quat_w   = _flat(cmd._all_body_quat_w)     # [NM, B, 4]
        self.all_body_lin_vel_w = _flat(cmd._all_body_lin_vel_w) # [NM, B, 3]
        self.all_body_ang_vel_w = _flat(cmd._all_body_ang_vel_w) # [NM, B, 3]

        self.motion_time_totals = cmd.motion_time_totals.to("cpu")  # [num_motions]
        self.num_motions = num_motions
        self.max_steps   = max_steps

    def forward(self, obs: torch.Tensor, motion_id: torch.Tensor, time_step: torch.Tensor):
        mid = motion_id.long().squeeze(-1)                           # [B]
        time_lim = self.motion_time_totals[mid] - 1                  # [B]
        t = torch.min(time_step.long().squeeze(-1), time_lim)        # [B], ONNX Min op

        flat_idx = mid * self.max_steps + t                          # [B], 1-D index
        return (
            self.actor(self.normalizer(obs)),
            self.all_joint_pos[flat_idx],
            self.all_joint_vel[flat_idx],
            self.all_body_pos_w[flat_idx],
            self.all_body_quat_w[flat_idx],
            self.all_body_lin_vel_w[flat_idx],
            self.all_body_ang_vel_w[flat_idx],
        )

    def export(self, path: str, filename: str) -> None:
        self.to("cpu")
        obs_dim = self.actor[0].in_features
        obs       = torch.zeros(1, obs_dim)
        motion_id = torch.zeros(1, 1, dtype=torch.long)
        time_step = torch.zeros(1, 1, dtype=torch.long)
        torch.onnx.export(
            self,
            (obs, motion_id, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "motion_id", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    """Attach training metadata to a saved ONNX model.

    For :class:`MultiMotionCommand`-based environments the following extra
    keys are added:

    * ``num_motions``         – number of motions in the library
    * ``motion_names``        – CSV of motion names
    * ``motion_files``        – CSV of motion file paths
    * ``motion_selector_type``– selector strategy used during training
    """
    onnx_path = os.path.join(path, filename)

    observation_names = env.observation_manager.active_terms["policy"]
    observation_history_lengths: list[int] = []

    if env.observation_manager.cfg.policy.history_length is not None:
        observation_history_lengths = [env.observation_manager.cfg.policy.history_length] * len(observation_names)
    else:
        for name in observation_names:
            term_cfg = env.observation_manager.cfg.policy.to_dict()[name]
            history_length = term_cfg["history_length"]
            observation_history_lengths.append(1 if history_length == 0 else history_length)

    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": observation_names,
        "observation_history_lengths": observation_history_lengths,
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,
        "body_names": env.command_manager.get_term("motion").cfg.body_names,
    }

    # Multi-motion extra metadata
    from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommand
    cmd = env.command_manager.get_term("motion")
    if isinstance(cmd, MultiMotionCommand):
        lib = cmd.motion_library
        metadata["num_motions"] = lib.num_motions
        metadata["motion_names"] = [lib.get_motion_info(i).motion_name for i in range(lib.num_motions)]
        metadata["motion_files"] = [lib.get_motion_info(i).motion_file for i in range(lib.num_motions)]
        metadata["motion_selector_type"] = cmd.cfg.motion_selector_type

    model = onnx.load(onnx_path)
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
