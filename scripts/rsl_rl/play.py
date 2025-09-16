# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import pathlib
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a trained RSL-RL policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during play.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of envs.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Env seed override.")
parser.add_argument("--registry_name", type=str, default=None,
                    help="W&B registry name for motions (e.g. user/motions/your_artifact:latest).")
parser.add_argument("--wandb_path", type=str, default=None,
                    help="W&B run path (e.g. user/proj/runs/abcdef). If provided, checkpoint is fetched from W&B.")
parser.add_argument("--motion_file", type=str, default=None,
                    help="Override motion.npz path explicitly (takes priority).")
parser.add_argument("--load_run", type=str, default=None, help="Load run name (when reading from logs/).")
parser.add_argument("--load_checkpoint", type=int, default=None, help="Checkpoint index (when reading from logs/).")
# 追加 RSL-RL 与 AppLauncher 参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# 录像时打开相机
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv 给 Hydra
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------
# 其余导入（在 app 启动后）
# ---------------------------------------------------------------------
import gymnasium as gym
import torch

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ⚠️ 与 train 对齐：使用自定义 Runner
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

# 可选导出
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _maybe_set_motion_from_registry(env_cfg, registry_name_cli):
    """对齐 train：优先从 registry 拉 motion.npz。"""
    if registry_name_cli is None:
        return False
    import wandb
    api = wandb.Api()
    name = registry_name_cli if ":" in registry_name_cli else (registry_name_cli + ":latest")
    art = api.artifact(name)
    env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    return True


def _maybe_set_motion_from_wandb_run(env_cfg, wandb_path):
    """如果提供了 wandb run，尝试从 used_artifacts 中找到 motions。"""
    import wandb
    api = wandb.Api()
    run = api.run(wandb_path)
    art = next((a for a in run.used_artifacts() if a.type == "motions"), None)
    if art is None:
        return False
    env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
    return True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlOnPolicyRunnerCfg):
    # 与 train 一致：用 cli_args 更新 agent_cfg；同步 num_envs / seed / device
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # --------- 保障 motion_file 一定被设置 ----------
    motion_set = False
    # 1) 显式提供 --motion_file：最高优先级
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        motion_set = True
    # 2) 提供 --registry_name：与 train 行为一致
    if not motion_set and args_cli.registry_name:
        motion_set = _maybe_set_motion_from_registry(env_cfg, args_cli.registry_name)
    # 3) 提供 --wandb_path：从该 run 的 used_artifacts 中找 motions
    if not motion_set and args_cli.wandb_path:
        motion_set = _maybe_set_motion_from_wandb_run(env_cfg, args_cli.wandb_path)

    if not motion_set:
        raise RuntimeError(
            "motion_file 未设置。请使用 --motion_file 或 --registry_name 或 --wandb_path 任一方式提供 motion.npz。"
        )

    # --------- 定位 checkpoint ----------
    if args_cli.wandb_path:
        # 从 W&B run 下载最大的 model_*.pt
        import wandb
        api = wandb.Api()
        run_path = args_cli.wandb_path
        run = api.run(run_path if "model" not in run_path else "/".join(run_path.split("/")[:-1]))
        files = [f.name for f in run.files() if "model" in f.name and f.name.endswith(".pt")]
        if not files:
            raise RuntimeError(f"W&B run `{run_path}` 中没有找到 model_*.pt")
        ckpt_file = (args_cli.wandb_path.split("/")[-1]
                     if "model" in args_cli.wandb_path else
                     max(files, key=lambda x: int(x.split("_")[1].split(".")[0])))
        run.file(ckpt_file).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{ckpt_file}"
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] Resume from W&B: {resume_path}")
    else:
        # 从本地 logs/rsl_rl/<experiment>/ 下查找
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run or args_cli.load_run,
                                          agent_cfg.load_checkpoint if agent_cfg.load_checkpoint is not None
                                          else args_cli.load_checkpoint)
        print(f"[INFO] Resume from logs: {resume_path}")
        log_dir = os.path.dirname(resume_path)

    # --------- 创建环境 ----------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # --------- 加载模型（与 train 同 Runner！） ----------
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # 导出（可选）
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)
    export_motion_policy_as_onnx(
        env.unwrapped,
        runner.alg.policy,
        normalizer=runner.obs_normalizer,
        path=export_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_dir)

    # --------- 推理回放 ----------
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    obs, _ = env.get_observations()
    steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            steps += 1
            if steps >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
