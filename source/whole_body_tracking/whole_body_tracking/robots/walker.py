# robot_isaac_config.py
# 可运行的 IsaacSim 风格的关节/致动器配置，基于你提供的关节名/电机编号。
# NOTE: 请把 URDF_PATH 改为你的 URDF 文件路径（或使用项目的 ASSET_DIR）。

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ---------- TODO: 改为你的 URDF 路径 ----------
URDF_PATH = "assets/walker_tienkung_ei_description/urdf/walker_tienkung_ei.urdf"

# ---------- 电机等效转动惯量（使用你原来样例的默认值，若有 datasheet 请替换） ----------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

# ---------- 目标固有频率 & 阻尼比（可以调） ----------
FREQ_HZ = 10.0  # 10 Hz 的自然频率（与你给的样例一致）
NATURAL_FREQ = 2.0 * 3.1415926535 * FREQ_HZ
DAMPING_RATIO = 2.0  # 样例使用较高阻尼（过阻尼），若需要更快响应可减小

# ---------- 根据公式计算刚度与阻尼 ----------
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

joint_to_motor = {
    "hip_roll_l_joint": 51,
    "hip_pitch_l_joint": 52,
    "hip_yaw_l_joint": 53,
    "knee_pitch_l_joint": 54,
    "ankle_pitch_l_joint": 55,
    "ankle_roll_l_joint": 56,
    "hip_roll_r_joint": 61,
    "hip_pitch_r_joint": 62,
    "hip_yaw_r_joint": 63,
    "knee_pitch_r_joint": 64,
    "ankle_pitch_r_joint": 65,
    "ankle_roll_r_joint": 66,
    "waist_yaw_joint": 31,
    "head_yaw_joint": 3,
    "head_pitch_joint": 2,
    "head_roll_joint": 1,
    "shoulder_pitch_l_joint": 11,
    "shoulder_roll_l_joint": 12,
    "shoulder_yaw_l_joint": 13,
    "elbow_pitch_l_joint": 14,
    "elbow_yaw_l_joint": 15,
    "wrist_pitch_l_joint": 16,
    "wrist_roll_l_joint": 17,
    "shoulder_pitch_r_joint": 21,
    "shoulder_roll_r_joint": 22,
    "shoulder_yaw_r_joint": 23,
    "elbow_pitch_r_joint": 24,
    "elbow_yaw_r_joint": 25,
    "wrist_pitch_r_joint": 26,
    "wrist_roll_r_joint": 27,
    # 如果还有 left arm elbow/wrist，请在这里补充
}

# ---------- 构建 ArticulationCfg（基于你最初示例结构） ----------
WALKER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            # TODO: 参考Tienkung-Lab中的配置，其余默认初始角度,按需改写为实际站立位姿
            "hip_roll_l_joint": -0.119,
            "hip_pitch_l_joint": -0.011,
            "hip_yaw_l_joint": -0.015,
            "knee_pitch_l_joint": 0.021,
            "ankle_pitch_l_joint": -0.019,
            "ankle_roll_l_joint": 0.074,
            "hip_roll_r_joint": -0.017,
            "hip_pitch_r_joint": 0.011,
            "hip_yaw_r_joint": -0.133,
            "knee_pitch_r_joint": 0.0,
            "ankle_pitch_r_joint": 0.088,
            "ankle_roll_r_joint": 0.095,
            "waist_yaw_joint": -0.162,
            "head_yaw_joint": -0.105,
            "head_pitch_joint": 0.0,
            "head_roll_joint": 0.0,
            "shoulder_pitch_l_joint": 0.2,
            "shoulder_roll_l_joint": 0.1,
            "shoulder_yaw_l_joint": -0.312,
            "elbow_pitch_l_joint": -0.301,
            "elbow_yaw_l_joint": 0.01,
            "wrist_pitch_l_joint": 0.231,
            "wrist_roll_l_joint": 0.0,
            "shoulder_pitch_r_joint": 0.15,
            "shoulder_roll_r_joint": -0.38,
            "shoulder_yaw_r_joint": 0.027,
            "elbow_pitch_r_joint": -0.293,
            "elbow_yaw_r_joint": -0.128,
            "wrist_pitch_r_joint": -0.196,
            "wrist_roll_r_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 左/右腿关节配置：mirrors 你原来样例（hip_yaw/pitch/roll, knee）
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_yaw_.*_joint",
                "hip_roll_.*_joint",
                "hip_pitch_.*_joint",
                "knee_pitch_.*_joint",
            ],
            effort_limit_sim={
                "hip_roll_.*_joint": 180,
                "hip_pitch_.*_joint": 300,
                "hip_yaw_.*_joint": 180,
                "knee_pitch_.*_joint": 300,
            },
            velocity_limit_sim={
                "hip_roll_.*_joint": 15.6,
                "hip_pitch_.*_joint": 15.6,
                "hip_yaw_.*_joint": 15.6,
                "knee_pitch_.*_joint": 15.6,
            },
            stiffness={
                "hip_roll_.*_joint": 700,
                "hip_pitch_.*_joint": 700,
                "hip_yaw_.*_joint": 500,
                "knee_pitch_.*_joint": 700,
            },
            damping={
                "hip_roll_.*_joint": 10,
                "hip_pitch_.*_joint": 10,
                "hip_yaw_.*_joint": 5,
                "knee_pitch_.*_joint": 10,
            },
            # armature={
            #     "hip_yaw_.*_joint": ARMATURE_7520_14,
            #     "hip_roll_.*_joint": ARMATURE_7520_22,
            #     "hip_pitch_.*_joint": ARMATURE_7520_14,
            #     "knee_pitch_.*_joint": ARMATURE_7520_22,
            # },
        ),

        # feet / ankle
        "feet": ImplicitActuatorCfg(
            # effort_limit_sim=50.0,
            # velocity_limit_sim=14.1,
            joint_names_expr=["ankle_pitch_.*_joint", "ankle_roll_.*_joint"],
            # stiffness=2.0 * STIFFNESS_5020,
            # damping=2.0 * DAMPING_5020,
            # armature=2.0 * ARMATURE_5020,

            effort_limit_sim={
                "ankle_pitch_.*_joint": 60,
                "ankle_roll_.*_joint": 30,
            },
            velocity_limit_sim={
                "ankle_pitch_.*_joint": 12.8,
                "ankle_roll_.*_joint": 7.8,
            },
            stiffness={
                "ankle_pitch_.*_joint": 30,
                "ankle_roll_.*_joint": 16.8,
            },
            damping={
                "ankle_pitch_.*_joint": 2.5,
                "ankle_roll_.*_joint": 1.4,
            },
        ),

        # waist yaw（如果你的腰只有 yaw）
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),

        "head": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["head_.*_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),

        # arms（这里以右臂关节名为例；左臂若有请在 joint_names_expr 添加）
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pitch_.*_joint",
                "shoulder_roll_.*_joint",
                "shoulder_yaw_.*_joint",
                "elbow_pitch_.*_joint",
                "elbow_yaw_.*_joint",
                "wrist_pitch_.*_joint",
                "wrist_roll_.*_joint",
            ],
            effort_limit_sim={
                "shoulder_pitch_.*_joint": 52.5,
                "shoulder_roll_.*_joint": 52.5,
                "shoulder_yaw_.*_joint": 52.5,
                "elbow_pitch_.*_joint": 52.5,
                "elbow_yaw_.*_joint": 52.5,
                "wrist_pitch_.*_joint": 52.5,
                "wrist_roll_.*_joint": 52.5,
            },
            velocity_limit_sim={
                "shoulder_pitch_.*_joint": 14.1,
                "shoulder_roll_.*_joint": 14.1,
                "shoulder_yaw_.*_joint": 14.1,
                "elbow_pitch_.*_joint": 14.1,
                "elbow_yaw_.*_joint": 14.1,
                "wrist_pitch_.*_joint": 14.1,
                "wrist_roll_.*_joint": 14.1,
            },
            stiffness={
                "shoulder_pitch_.*_joint": 60,
                "shoulder_roll_.*_joint": 20,
                "shoulder_yaw_.*_joint": 10,
                "elbow_pitch_.*_joint": 10,
                "elbow_yaw_.*_joint": 10,
                "wrist_pitch_.*_joint": 10,
                "wrist_roll_.*_joint": 10,
            },
            damping={
                "shoulder_pitch_.*_joint": 3,
                "shoulder_roll_.*_joint": 1.5,
                "shoulder_yaw_.*_joint": 1,
                "elbow_pitch_.*_joint": 1,
                "elbow_yaw_.*_joint": 1,
                "wrist_pitch_.*_joint": 1,
                "wrist_roll_.*_joint": 1,
            },
            # armature={
            #     "shoulder_pitch_.*_joint": ARMATURE_5020,
            #     "shoulder_roll_.*_joint": ARMATURE_5020,
            #     "shoulder_yaw_.*_joint": ARMATURE_5020,
            #     "elbow_pitch_.*_joint": ARMATURE_5020,
            #     "elbow_yaw_.*_joint": ARMATURE_5020,
            #     "wrist_pitch_.*_joint": ARMATURE_5020,
            #     "wrist_roll_.*_joint": ARMATURE_4010,
            # },
        ),
    },
)

# ---------- 生成 ACTION_SCALE（动作缩放系数，供 controller / RL 使用） ----------
WALKER_ACTION_SCALE = {}
for a in WALKER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    # normalize scalars into dicts keyed by the regex string:
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        # NOTE: n here is a regex pattern string (as used above). We'll keep the same key.
        if n in e and n in s and s[n]:
            WALKER_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

# ---------- 导出 / 打印一些便于检查的信息 ----------
# def print_config_summary():
#     print("URDF_PATH =", URDF_PATH)
#     print("Joint -> motor mapping (sample):")
#     for j, m in joint_to_motor.items():
#         print(f"  {j} -> motor {m}")
#     print("\nExample ACTION_SCALE entries (regex -> scale):")
#     for k, v in list(ACTION_SCALE.items())[:10]:
#         print(f"  {k} : {v}")
#
# if __name__ == "__main__":
#     # 简单检查：打印 summary
#     print("Loaded robot ISAAC config. Summary:")
#     print("NATURAL_FREQ (rad/s):", NATURAL_FREQ)
#     print("DAMPING_RATIO:", DAMPING_RATIO)
#     print("Armature constants:")
#     print("  ARMATURE_5020", ARMATURE_5020)
#     print("  ARMATURE_7520_14", ARMATURE_7520_14)
#     print("  ARMATURE_7520_22", ARMATURE_7520_22)
#     print("  ARMATURE_4010", ARMATURE_4010)
#     print("\nActuator groups:", list(WALKER_CFG.actuators.keys()))
#     print("\nSample joint_to_motor mapping (first 10):")
#     for i, (j, m) in enumerate(joint_to_motor.items()):
#         print(f"  {j} -> {m}")
#         if i >= 9:
#             break
#     print("\nSample ACTION_SCALE (first 10 regex keys):")
#     for i, (k, v) in enumerate(ACTION_SCALE.items()):
#         print(f"  {k} -> {v}")
#         if i >= 9:
#             break
