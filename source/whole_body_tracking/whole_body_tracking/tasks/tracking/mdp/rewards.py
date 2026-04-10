from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


# Multi-motion aware reward functions
def motion_difficulty_scaling(
    env: ManagerBasedRLEnv, command_name: str, base_reward: float = 1.0
) -> torch.Tensor:
    """Scale reward based on motion difficulty.

    Environments executing more difficult motions get lower baseline rewards,
    encouraging the policy to focus on easier motions first during curriculum learning.

    Args:
        env: Environment instance
        command_name: Name of the command term
        base_reward: Base reward value before scaling (default: 1.0)

    Returns:
        Tensor of shape (num_envs,) with difficulty-scaled rewards
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)

    # Get difficulty level for each environment's current motion
    difficulties = []
    for motion_id in command.current_motion_ids:
        info = command.motion_library.get_motion_info(int(motion_id))
        difficulties.append(info.difficulty_level)

    difficulties = torch.tensor(difficulties, dtype=torch.float32, device=env.device)

    # Reward = base_reward * (1 - 0.5 * difficulty)
    # Easy motions (difficulty=0) get full reward
    # Hard motions (difficulty=1) get 50% reward
    scaling = 1.0 - 0.5 * difficulties
    return base_reward * scaling


def motion_diversity_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    weight: float = 0.01,
) -> torch.Tensor:
    """Bonus reward for exploring diverse motions.

    Encourages the policy to learn all motions rather than converging to a subset.
    This is most useful during early training stages.

    Args:
        env: Environment instance
        command_name: Name of the command term
        weight: Weight of the diversity bonus (default: 0.01)

    Returns:
        Tensor of shape (num_envs,) with diversity bonus
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)

    # Count how many unique motions are being executed across all environments
    unique_motions = len(torch.unique(command.current_motion_ids))
    total_motions = command.motion_library.num_motions

    # Diversity ratio: 0 to 1
    diversity_ratio = unique_motions / max(total_motions, 1)

    # Apply bonus: all environments get same bonus based on diversity
    # This encourages natural exploration of motions
    return weight * diversity_ratio * torch.ones(env.num_envs, dtype=torch.float32, device=env.device)


def motion_switching_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    weight: float = -0.01,
) -> torch.Tensor:
    """Penalty for frequently switching between motions.

    Encourages the policy to spend longer time on each motion before switching,
    promoting more stable learning.

    Args:
        env: Environment instance
        command_name: Name of the command term
        weight: Weight of the penalty (default: -0.01, should be negative)

    Returns:
        Tensor of shape (num_envs,) with switching penalty
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)

    # Penalty is proportional to motion change count
    # More changes = larger penalty (more negative reward)
    penalty = weight * command.motion_change_counts.float()

    return penalty

