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

##
# Scene definition
##

# 机器人速度扰动范围（用于域随机化）
# 单位: 线速度(m/s), 角速度(rad/s)
VELOCITY_RANGE = {
    "x": (-0.5, 0.5),      # 前后方向线速度
    "y": (-0.5, 0.5),      # 左右方向线速度
    "z": (-0.2, 0.2),      # 上下方向线速度
    "roll": (-0.52, 0.52), # 翻滚角速度(约30度/秒)
    "pitch": (-0.52, 0.52),# 俯仰角速度(约30度/秒)
    "yaw": (-0.78, 0.78),  # 偏航角速度(约45度/秒)
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # 地面地形配置
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",          # USD场景中的路径
        terrain_type="plane",              # 平面地形类型
        collision_group=-1,                # 碰撞组ID(-1表示与所有组碰撞)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 摩擦力组合模式
            restitution_combine_mode="multiply", # 弹性恢复组合模式
            static_friction=1.0,           # 静摩擦系数
            dynamic_friction=1.0,          # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,              # 启用UV投影
        ),
    )
    # 机器人配置（将在具体任务中指定）
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),      # 光源颜色(RGB)
            intensity=3000.0               # 光照强度
        ),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.13, 0.13, 0.13),      # 环境光颜色
            intensity=1000.0               # 环境光强度
        ),
    )

    # 接触力传感器配置
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", # 监测机器人所有部件的接触
        history_length=3,                   # 保存3帧历史数据
        track_air_time=True,                # 跟踪腾空时间
        force_threshold=10.0,               # 接触力阈值(N)
        debug_vis=True                      # 启用调试可视化
    )


##
# MDP设置 (马尔可夫决策过程)
##


@configclass
class CommandsCfg:
    """MDP命令规范配置

    定义机器人需要跟踪的运动命令，包括参考动作的采样和随机化参数。
    """

    motion = mdp.MotionCommandCfg(
        asset_name="robot",                    # 目标资产名称
        resampling_time_range=(1.0e9, 1.0e9), # 重采样时间范围(s) - 极大值表示不重采样
        debug_vis=True,                       # 启用调试可视化
        pose_range={                          # 姿态随机化范围
            "x": (-0.05, 0.05),              # X方向位置偏移(m)
            "y": (-0.05, 0.05),              # Y方向位置偏移(m)
            "z": (-0.01, 0.01),              # Z方向位置偏移(m)
            "roll": (-0.1, 0.1),             # 翻滚角偏移(rad)
            "pitch": (-0.1, 0.1),            # 俯仰角偏移(rad)
            "yaw": (-0.2, 0.2),              # 偏航角偏移(rad)
        },
        velocity_range=VELOCITY_RANGE,        # 速度随机化范围
        joint_position_range=(-0.1, 0.1),    # 关节位置随机化范围(rad)
    )


@configclass
class ActionsCfg:
    """MDP动作规范配置

    定义机器人的控制动作空间，这里使用关节位置控制。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",           # 目标机器人资产
        joint_names=[".*"],          # 控制所有关节(正则表达式)
        use_default_offset=True      # 使用默认关节位置作为偏移
    )


@configclass
class ObservationsCfg:
    """MDP观察规范配置

    定义策略网络和评论家网络的观察空间。策略观察包含噪声以提高sim-to-real转移性能。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略网络观察组配置

        包含有噪声的观察，用于训练鲁棒的策略。观察项顺序会被保持。
        """

        # 观察项定义（保持顺序）
        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"}
        )  # 运动命令

        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.25, n_max=0.25)  # 锚点位置噪声
        )

        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=-0.05, n_max=0.05)  # 锚点方向噪声
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.5, n_max=0.5)    # 基座线速度噪声
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2)    # 基座角速度噪声
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01)  # 关节位置噪声
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5)    # 关节速度噪声
        )

        actions = ObsTerm(func=mdp.last_action)  # 上一步动作

        def __post_init__(self):
            self.enable_corruption = True      # 启用观察破坏（噪声）
            self.concatenate_terms = True      # 将所有观察项连接为一个向量

    @configclass
    class PrivilegedCfg(ObsGroup):
        """特权观察组配置（评论家网络）

        包含无噪声的精确观察，以及额外的特权信息（如身体位置/方向）。
        用于训练评论家网络进行价值函数估计。
        """

        command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion"}
        )  # 运动命令（无噪声）

        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"}
        )  # 锚点位置（无噪声）

        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"}
        )  # 锚点方向（无噪声）

        body_pos = ObsTerm(
            func=mdp.robot_body_pos_b,
            params={"command_name": "motion"}
        )  # 机器人身体位置（特权信息）

        body_ori = ObsTerm(
            func=mdp.robot_body_ori_b,
            params={"command_name": "motion"}
        )  # 机器人身体方向（特权信息）

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)     # 基座线速度（无噪声）
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)     # 基座角速度（无噪声）
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)       # 关节位置（无噪声）
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)       # 关节速度（无噪声）
        actions = ObsTerm(func=mdp.last_action)           # 上一步动作

    # 观察组实例
    policy: PolicyCfg = PolicyCfg()        # 策略网络观察
    critic: PrivilegedCfg = PrivilegedCfg() # 评论家网络观察


@configclass
class EventCfg:
    """Configuration for events."""

    # 启动时事件
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",                    # 在每个episode开始时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),    # 静摩擦系数范围
            "dynamic_friction_range": (0.3, 1.2),   # 动摩擦系数范围
            "restitution_range": (0.0, 0.5),        # 弹性恢复系数范围
            "num_buckets": 64,                      # 随机化桶数量
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",                              # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01), # 关节默认位置随机化范围(rad)
            "operation": "add",                       # 加法操作
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",                              # 启动时执行
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"), # 修改body_names对应机器人的身体部位
            "com_range": {                           # 质心随机化范围(m)
                "x": (-0.025, 0.025),               # X方向质心偏移
                "y": (-0.05, 0.05),                 # Y方向质心偏移
                "z": (-0.05, 0.05)                  # Z方向质心偏移
            },
        },
    )

    # 间隔事件 - 传统推动（对所有环境）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",                         # 间隔执行模式
        interval_range_s=(1.0, 3.0),           # 执行间隔范围(s)
        params={"velocity_range": VELOCITY_RANGE}, # 推力速度范围
    )

@configclass
class RewardsCfg:
    """MDP奖励项配置

    定义运动跟踪任务的奖励函数，包括位置、方向、速度跟踪奖励和行为惩罚。
    使用指数衰减奖励函数以实现精确跟踪。
    """

    # DeepMimic风格的运动跟踪奖励
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,                           # 奖励权重
        params={
            "command_name": "motion",
            "std": 0.3                        # 标准差参数，控制奖励衰减速度
        },
    )  # 全局锚点位置跟踪奖励

    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "std": 0.4
        },
    )  # 全局锚点方向跟踪奖励

    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.3
        },
    )  # 相对身体位置跟踪奖励

    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.4
        },
    )  # 相对身体方向跟踪奖励

    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 1.0
        },
    )  # 全局身体线速度跟踪奖励

    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 3.14
        },
    )  # 全局身体角速度跟踪奖励
    # 行为正则化惩罚项
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-1e-1                          # 负权重表示惩罚
    )  # 动作变化率L2惩罚，鼓励平滑动作

    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,                         # 强惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])
        },
    )  # 关节限位惩罚，防止关节超出安全范围

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,                          # 接触惩罚权重
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    # 正则表达式：排除脚踝和手腕的所有body
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,                 # 接触力阈值(N)
        },
    )  # 不期望接触惩罚，避免非末端执行器接触地面

@configclass
class TerminationsCfg:
    """MDP终止条件配置

    定义episode提前结束的条件，用于避免危险状态和无效训练数据。
    """

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True                         # 标记为超时终止
    )  # 时间超时终止

    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25                 # Z方向位置偏差阈值(m)
        },
    )  # 锚点位置偏差过大终止（仅检查Z方向）

    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "motion",
            "threshold": 0.8                  # 方向偏差阈值
        },
    )  # 锚点方向偏差过大终止（防止机器人倾倒）

    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,                # 末端执行器位置偏差阈值(m)
            "body_names": [                   # 监控的末端执行器
                "left_ankle_roll_link",       # 左脚踝
                "right_ankle_roll_link",      # 右脚踝
                "left_wrist_yaw_link",        # 左手腕
                "right_wrist_yaw_link",       # 右手腕
            ],
        },
    )  # 末端执行器位置偏差过大终止

@configclass
class CurriculumCfg:
    """MDP课程学习配置

    定义训练过程中的课程学习策略，可以逐步增加任务难度。
    包含力课程学习等自适应训练策略。
    """

    # 力课程学习项 - 根据机器人表现动态调整辅助力
    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """运动跟踪环境配置

    整合所有MDP组件的完整环境配置，用于训练人形机器人跟踪参考运动。
    基于Isaac Lab的ManagerBasedRLEnv框架。
    """

    # 场景设置
    scene: MySceneCfg = MySceneCfg(
        num_envs=4096,                        # 并行环境数量
        env_spacing=2.5                      # 环境间距(m)
    )

    # 基础MDP组件
    observations: ObservationsCfg = ObservationsCfg()  # 观察空间配置
    actions: ActionsCfg = ActionsCfg()                  # 动作空间配置
    commands: CommandsCfg = CommandsCfg()               # 命令配置

    # MDP行为定义
    rewards: RewardsCfg = RewardsCfg()                  # 奖励函数配置
    terminations: TerminationsCfg = TerminationsCfg()   # 终止条件配置
    events: EventCfg = EventCfg()                       # 随机化事件配置
    curriculum: CurriculumCfg = CurriculumCfg()         # 课程学习配置

    def __post_init__(self):
        """后初始化配置

        设置仿真参数、渲染设置和查看器配置。
        """
        # 通用设置
        self.decimation = 4                   # 控制频率抽取率 (仿真50Hz → 控制12.5Hz)
        self.episode_length_s = 10.0          # Episode时长(s)

        # 仿真设置
        self.sim.dt = 0.005                   # 仿真时间步长(s) = 200Hz
        self.sim.render_interval = self.decimation  # 渲染间隔
        self.sim.physics_material = self.scene.terrain.physics_material  # 物理材质
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15  # GPU刚体patch最大数量

        # 查看器设置
        self.viewer.eye = (1.5, 1.5, 1.5)    # 相机位置
        self.viewer.origin_type = "asset_root" # 相机原点类型
        self.viewer.asset_name = "robot"      # 跟随的资产名称
