from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import whole_body_tracking.tasks.tracking.mdp as mdp

from whole_body_tracking.robots.tienkung import TIENKUNG_ACTION_SCALE, TIENKUNG_CFG
from whole_body_tracking.tasks.tracking.config.walker.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg



@configclass
class TienkungFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = TIENKUNG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = TIENKUNG_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            'hip_roll_l_link',
            'hip_pitch_l_link',
            'hip_yaw_l_link',
            'knee_pitch_l_link',
            'ankle_pitch_l_link',
            'ankle_roll_l_link',

            'hip_roll_r_link',
            'hip_pitch_r_link',
            'hip_yaw_r_link',
            'knee_pitch_r_link',
            'ankle_pitch_r_link',
            'ankle_roll_r_link',

            ## arm
            'shoulder_pitch_l_link',
            'shoulder_roll_l_link',
            'shoulder_yaw_l_link',
            'elbow_pitch_l_link',

            'shoulder_pitch_r_link',
            'shoulder_roll_r_link',
            'shoulder_yaw_r_link',
            'elbow_pitch_r_link',
        ]

        # 相机设置：自由视角，不跟随机器人
        self.viewer.eye = (3.0, 3.0, 2.0)  # 相机位置
        self.viewer.lookat = (0.0, 0.0, 1.0)  # 看向位置
        self.viewer.origin_type = "world"  # 世界坐标系，不跟随机器人
        self.viewer.asset_name = None  # 不绑定到特定资产

        # 关闭调试可视化显示
        self.commands.motion.debug_vis = False  # 关闭motion命令的调试可视化
        self.scene.contact_forces.debug_vis = False  # 关闭接触力可视化



@configclass
class TienkungFlatWoStateEstimationEnvCfg(TienkungFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class TienkungFlatLowFreqEnvCfg(TienkungFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
