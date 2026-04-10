from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


# Multi-motion observation functions
def motion_id_encoding(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """One-hot encoding of current motion ID for multi-motion training.

    Returns a one-hot vector indicating which motion each environment is currently executing.
    This encoding allows the policy to distinguish between different reference motions.

    Args:
        env: Environment instance
        command_name: Name of the command term

    Returns:
        Tensor of shape (num_envs, num_motions) with one-hot encoding
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    num_motions = command.motion_library.num_motions

    # Create one-hot encoding
    one_hot = torch.zeros(env.num_envs, num_motions, dtype=torch.float32, device=env.device)
    one_hot[torch.arange(env.num_envs, device=env.device), command.current_motion_ids] = 1.0

    return one_hot


def motion_change_signal(env: ManagerBasedEnv, command_name: str, window_size: int = 5) -> torch.Tensor:
    """Signal indicating whether motion has recently changed.

    Returns 1.0 for the first 'window_size' steps after a motion change, 0.0 otherwise.
    This helps the policy adapt to motion transitions.

    Args:
        env: Environment instance
        command_name: Name of the command term
        window_size: Number of steps to signal after motion change (default: 5)

    Returns:
        Tensor of shape (num_envs, 1) with motion change signal
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)

    # Signal is 1.0 for first window_size steps, 0.0 otherwise
    signal = (command.time_steps < window_size).float().view(-1, 1)

    return signal


def motion_progress(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Normalized progress through current motion (0.0 to 1.0).

    Indicates how far through the current motion execution we are.
    Helps the policy anticipate motion ends and prepare for resampling.

    Args:
        env: Environment instance
        command_name: Name of the command term

    Returns:
        Tensor of shape (num_envs, 1) with normalized progress [0.0, 1.0]
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)

    # Vectorized: use pre-built motion_time_totals tensor (no Python loop)
    max_steps = command.motion_time_totals[command.current_motion_ids].float().view(-1, 1)
    progress = (command.time_steps.float().view(-1, 1) + 1.0) / (max_steps + 1e-6)
    return torch.clamp(progress, 0.0, 1.0)


def motion_id_embedding(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Continuous embedding for the current motion ID.

    Returns a fixed-size vector from the ``MotionEmbeddingBank`` stored in the
    command term.  Preferred over ``motion_id_encoding`` (one-hot) when the
    motion library has >= 50 motions because embedding_dim << num_motions.

    Raises ``RuntimeError`` if the command term does not have an embedding bank
    (i.e. ``use_embedding`` was False / auto-disabled).

    Args:
        env: Environment instance
        command_name: Name of the multi-motion command term

    Returns:
        Tensor of shape (num_envs, embedding_dim)
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)

    if command.embedding_bank is None:
        raise RuntimeError(
            f"motion_id_embedding: command '{command_name}' has no embedding bank. "
            "Set use_embedding=True (or num_motions >= 50) in MultiMotionCommandCfg."
        )

    return command.embedding_bank.get(command.current_motion_ids)

