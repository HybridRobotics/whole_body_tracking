from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import RewardsCfg, TerminationsCfg, \
    EventCfg


@configclass
class TienkungEventCfg(EventCfg):
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",                              # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"), # 修改body_names对应机器人的身体部位
            "com_range": {                           # 质心随机化范围(m)
                "x": (-0.025, 0.025),               # X方向质心偏移
                "y": (-0.05, 0.05),                 # Y方向质心偏移
                "z": (-0.05, 0.05)                  # Z方向质心偏移
            },
        },
    )


@configclass
class TienkungRewardsCfg(RewardsCfg):
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,  # 接触惩罚权重
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    # 正则表达式：排除脚踝和手腕的所有body
                    r"^(?!ankle_roll_l_link$)(?!ankle_roll_r_link$)(?!elbow_pitch_l_link)(?!elbow_pitch_r_link).+$"
                ],
            ),
            "threshold": 1.0,  # 接触力阈值(N)
        },
    )  # 不期望接触惩罚，避免非末端执行器接触地面


@configclass
class TienkungTerminationsCfg(TerminationsCfg):
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.3,                # 末端执行器位置偏差阈值(m)
            "body_names": [                   # 监控的末端执行器
                "ankle_roll_l_link",       # 左脚踝
                "ankle_roll_r_link",      # 右脚踝
                "elbow_pitch_l_link",        # 左手腕
                "elbow_pitch_r_link",       # 右手腕
            ],
        },
    )  # 末端执行器位置偏差过大终止



##
# Environment configuration
##


@configclass
class TienkungTrackingEnvCfg(TrackingEnvCfg):

    # MDP行为定义
    rewards: TienkungRewardsCfg = TienkungRewardsCfg()                  # 奖励函数配置
    terminations: TienkungTerminationsCfg = TienkungTerminationsCfg()   # 终止条件配置
    events: TienkungEventCfg = TienkungEventCfg()                       # 随机化事件配置
