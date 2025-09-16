import os
from pathlib import Path
from typing import Optional

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

try:
    import wandb  # 可用就用；不可用时走兜底
except Exception:  # pragma: no cover
    wandb = None

from whole_body_tracking.utils.exporter import (
    attach_onnx_metadata,
    export_motion_policy_as_onnx,
)


def _dirname_and_runname_from_checkpoint(path: str) -> tuple[str, str]:
    """
    path: 形如 ".../logs/.../<run_name>/model_XXXX.pt"
    返回: (dir_path, run_name)
    """
    dir_path = os.path.dirname(os.path.abspath(path))
    run_name = os.path.basename(dir_path)
    return dir_path, run_name


class _NormalizerMixin:
    """在多个常见位置/命名里尝试获取观测归一化器。找不到则返回 None。"""

    def _get_obs_normalizer(self):
        alg = getattr(self, "alg", None)
        vec_env = getattr(self, "vec_env", None)
        env = getattr(self, "env", None)

        candidates = [
            getattr(self, "obs_normalizer", None),
            getattr(self, "obs_norm", None),
            getattr(self, "obs_rms", None),
            getattr(alg, "obs_normalizer", None) if alg else None,
            getattr(alg, "obs_norm", None) if alg else None,
            getattr(alg, "obs_rms", None) if alg else None,
            getattr(vec_env, "obs_rms", None) if vec_env else None,
            getattr(env, "obs_rms", None) if env else None,
        ]
        for c in candidates:
            if c is not None:
                return c
        return None

    def _use_wandb(self) -> bool:
        return (
            getattr(self, "logger_type", None) == "wandb"
            and (wandb is not None)
            and (getattr(wandb, "run", None) is not None)
        )


class MyOnPolicyRunner(_NormalizerMixin, OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)

        if not self._use_wandb():
            return

        policy_dir, run_name = _dirname_and_runname_from_checkpoint(path)
        filename = f"{run_name}.onnx"
        normalizer = self._get_obs_normalizer()

        try:
            export_policy_as_onnx(
                self.alg.policy,
                normalizer=normalizer,  # 允许为 None
                path=policy_dir,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_dir, filename=filename)
            # 注意要 join，不要直接拼字符串
            wandb.save(os.path.join(policy_dir, filename), base_path=os.path.dirname(policy_dir))
        except Exception as e:  # 不让导出失败把训练中断
            print(f"[WARN] ONNX 导出或 W&B 保存失败: {e}")


class MotionOnPolicyRunner(_NormalizerMixin, OnPolicyRunner):
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: Optional[str] = None,
        device: str = "cpu",
        registry_name: Optional[str] = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)

        if not self._use_wandb():
            return

        policy_dir, run_name = _dirname_and_runname_from_checkpoint(path)
        filename = f"{run_name}.onnx"
        normalizer = self._get_obs_normalizer()

        try:
            export_motion_policy_as_onnx(
                self.env.unwrapped,
                self.alg.policy,
                normalizer=normalizer,  # 允许为 None
                path=policy_dir,
                filename=filename,
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_dir, filename=filename)
            wandb.save(os.path.join(policy_dir, filename), base_path=os.path.dirname(policy_dir))

            # 将本次运行与（可选的）artifact registry 关联一下
            if self.registry_name:
                try:
                    wandb.run.use_artifact(self.registry_name)
                except Exception as e:
                    print(f"[WARN] 关联 Artifact 失败（{self.registry_name}）: {e}")
                finally:
                    self.registry_name = None  # 只关联一次
        except Exception as e:
            print(f"[WARN] Motion ONNX 导出或 W&B 保存失败: {e}")
