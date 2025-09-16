"""Replay a motion from CSV and save as NPZ (local by default; optional W&B upload)

Usage examples:

# 只转本地（推荐）
python csv_to_npz.py \
  --input_file LAFAN/dance1_subject2.csv --input_fps 30 --frame_range 122 722 \
  --output_file ./motions/dance1_subject2.npz --output_fps 50 --headless

# 转本地 + 上传到 W&B 项目（不关联 registry）
python csv_to_npz.py \
  --input_file LAFAN/dance1_subject2.csv --output_file ./motions/dance1_subject2.npz \
  --upload_wandb --log_project_name csv_to_npz --run_name dance1_subject2 --headless

# 转本地 + 上传到 W&B 并尝试关联 registry（只有你已在 UI 里创建好自定义 registry 才能成功）
python csv_to_npz.py \
  --input_file LAFAN/dance1_subject2.csv --output_file ./motions/dance1_subject2.npz \
  --upload_wandb --log_project_name csv_to_npz --run_name dance1_subject2 \
  --registry_name xboninglix-technical-university-of-munich-org/wandb-registry-motions/dance1_subject2 \
  --headless
"""

import argparse
import os
import pathlib
import numpy as np

from isaaclab.app import AppLauncher

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input motion CSV.")
parser.add_argument("--input_fps", type=int, default=30, help="FPS of the input motion (CSV).")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=("Frame range START END (inclusive), 1-based index. If not set, load all frames."),
)
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output NPZ.")
parser.add_argument("--output_fps", type=int, default=50, help="FPS of the output motion.")
# 可选：上传到 W&B
parser.add_argument("--upload_wandb", action="store_true", help="Upload the saved NPZ to Weights & Biases.")
parser.add_argument("--log_project_name", type=str, default="csv_to_npz", help="W&B project name.")
parser.add_argument("--run_name", type=str, default=None, help="W&B run name (default: stem of output_file).")
parser.add_argument(
    "--registry_name",
    type=str,
    default=None,
    help=("Optional W&B registry path to link, e.g. "
          "xboninglix-technical-university-of-munich-org/wandb-registry-motions/dance1_subject2 "
          "(requires registry pre-created in UI)."),
)

# 让 AppLauncher 添加它自己的参数（例如 --headless, --renderer 等）
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------
# Imports 依赖 IsaacLab
# -------------------------
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

# 机器人配置
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file."""
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    self.motion_file,
                    delimiter=",",
                    skiprows=self.frame_range[0] - 1,
                    max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7]
        # convert to wxyz
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]

        self.input_frames = motion.shape[0]
        self.duration = max((self.input_frames - 1) * self.input_dt, 1e-6)
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration:.4f}s, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames} @ {self.input_fps}Hz -> "
            f"output frames: {self.output_frames} @ {self.output_fps}Hz"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1, device=times.device))
        blend = phase * (self.input_frames - 1) - idx0
        return idx0, idx1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Derivative on SO(3) using finite difference of relative rotations."""
        if rotations.shape[0] < 3:
            return torch.zeros((rotations.shape[0], 3), device=rotations.device, dtype=rotations.dtype)
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, joint_names: list[str]):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=tuple(args_cli.frame_range) if args_cli.frame_range else None,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # 仅渲染，不步进物理
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # 采样并缓存一遍序列
        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            # ---- 保存到本地 ----
            out_path = pathlib.Path(args_cli.output_file).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **log)
            print(f"[INFO]: Motion saved locally: {out_path}")

            # ---- 可选：上传到 W&B（不再强制）----
            if args_cli.upload_wandb:
                try:
                    import wandb
                    run_name = args_cli.run_name or out_path.stem
                    run = wandb.init(project=args_cli.log_project_name, name=run_name)
                    print(f"[INFO]: Logging motion to W&B: {run_name}")
                    logged_artifact = run.log_artifact(
                        artifact_or_path=str(out_path), name=run_name, type="motions"
                    )
                    if args_cli.registry_name:
                        # 只有在 UI 里先建好自定义 registry 才能成功；否则会 404
                        run.link_artifact(artifact=logged_artifact, target_path=args_cli.registry_name)
                        print(f"[INFO]: Linked to registry: {args_cli.registry_name}")
                except Exception as e:
                    print(f"[WARN]: W&B upload/link skipped due to error: {e}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Ready
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(
        sim,
        scene,
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
