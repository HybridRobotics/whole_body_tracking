from fsspec import entry_points
import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# 用于训练的环境（保持自适应采样）
gym.register(
    id="Tracking-Flat-Tienkung-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.TienkungFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TienkungFlatPPORunnerCfg",
    },
)
