# USAGE.md — Whole Body Tracking 使用文档

本文档涵盖从运动数据准备、单运动训练、多运动基础模型训练到部署推理的完整流程。

---

## 目录

1. [环境安装](#1-环境安装)
2. [运动数据准备](#2-运动数据准备)
3. [单运动训练（基础用法）](#3-单运动训练基础用法)
4. [回放与评估](#4-回放与评估)
5. [多运动基础模型训练（Phase 2/3）](#5-多运动基础模型训练phase-23)
6. [配置参考](#6-配置参考)
7. [ONNX 导出与部署](#7-onnx-导出与部署)
8. [性能分析](#8-性能分析)
9. [扩展：添加新机器人或新任务](#9-扩展添加新机器人或新任务)

---

## 1. 环境安装

```bash
# 激活 Isaac Lab 环境
conda activate env_isaaclab

# 安装本扩展包（开发模式）
pip install -e source/whole_body_tracking
```

验证安装：

```bash
python -c "import whole_body_tracking; print('OK')"
```

---

## 2. 运动数据准备

训练需要 `.npz` 格式的运动捕捉文件。源数据为 `.csv` 格式（关节角度 + 身体位置/姿态的时序数据）。

### 2.1 单文件转换

```bash
python scripts/csv_to_npz.py \
    --input_file  source/motion/walker/csv/walk_forward.csv \
    --input_fps   30 \
    --output_name source/motion/walker/npz/walk_forward \
    --output_fps  50 \
    --robot       walker \
    --no_wandb \
    --save_to     source/motion/walker/npz/
```

| 参数 | 说明 |
|------|------|
| `--input_file` | 输入 CSV 路径 |
| `--input_fps` | 原始采样频率（默认 30 Hz） |
| `--frame_range START END` | 截取帧范围（1-indexed，可选） |
| `--output_name` | 输出文件基名（无需 `.npz` 后缀） |
| `--output_fps` | 重采样到目标频率（默认 50 Hz） |
| `--robot` | 目标机器人：`tienkung` / `unitree_g1` / `walker` |
| `--no_wandb` | 不上传 WandB，直接保存到本地 |
| `--save_to` | 本地保存目录（默认 `/tmp/`） |

### 2.2 批量转换

递归搜索 `--input_dir` 下所有子目录中的 `.csv` 文件，并在 `--output_dir` 中保留相同的子目录结构。

```bash
python scripts/csv_to_npz_batch.py \
    --input_dir  source/motion/walker/csv/ \
    --output_dir source/motion/walker/npz/ \
    --robot      walker \
    --no_wandb
```

| 参数 | 说明 |
|------|------|
| `--input_dir` | 输入根目录（递归搜索所有子目录中的 `.csv`） |
| `--output_dir` | 输出根目录（自动镜像子目录结构） |
| `--robot` | 同 `csv_to_npz.py`，支持 `tienkung` / `unitree_g1` / `walker` |
| `--input_fps` | 原始采样频率（默认 30 Hz） |
| `--no_wandb` | 不上传 WandB，直接保存到本地 |

### 2.3 NPZ 文件格式

每个 `.npz` 文件包含以下键：

| 键 | 形状 | 说明 |
|----|------|------|
| `fps` | scalar | 帧率（Hz） |
| `joint_pos` | `[T, n_joints]` | 关节角度（rad） |
| `joint_vel` | `[T, n_joints]` | 关节速度（rad/s） |
| `body_pos_w` | `[T, n_bodies, 3]` | 身体位置（世界系，m） |
| `body_quat_w` | `[T, n_bodies, 4]` | 身体姿态四元数（世界系，wxyz）|
| `body_lin_vel_w` | `[T, n_bodies, 3]` | 线速度（世界系，m/s） |
| `body_ang_vel_w` | `[T, n_bodies, 3]` | 角速度（世界系，rad/s） |

其中 `T` 为总帧数，`n_joints` / `n_bodies` 取决于机器人。

---

## 3. 单运动训练（基础用法）

### 3.1 启动训练

```bash
python scripts/rsl_rl/train.py \
    --task       Tracking-Flat-Walker-v0 \
    --motion_file source/motion/walker/npz/walk_forward.npz \
    --num_envs   4096 \
    --headless
```

常用可选参数：

```bash
# 指定随机种子
--seed 42

# 限制训练迭代数
--max_iterations 10000

# 从上一次运行恢复训练
--resume
--load_run    2024-01-01_12-00-00_walker_flat
--load_checkpoint model_5000.pt

# 录制训练视频
--video --video_length 200 --video_interval 2000

# 使用 WandB 记录（需提前 wandb login）
--logger wandb --log_project_name my_project
```

### 3.2 已注册的任务

| 任务名 | 说明 |
|--------|------|
| `Tracking-Flat-Walker-v0` | Walker 机器人平地全身追踪 |
| `Tracking-Flat-Walker-WoSE-v0` | 同上，去除状态估计观测（`base_lin_vel` 和 `motion_anchor_pos_b`） |
| `Tracking-Flat-Walker-LowFreq-v0` | 低控制频率版本（适用于部署延迟较高的场景） |

### 3.3 训练日志

日志默认保存在：

```
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
├── params/
│   ├── env.yaml       # 环境配置快照
│   └── agent.yaml     # PPO 配置快照
├── model_<iter>.pt    # 检查点
└── *.onnx             # 自动导出的策略（WandB 模式下）
```

---

## 4. 回放与评估

### 4.1 可视化策略

```bash
python scripts/rsl_rl/play.py \
    --task        Tracking-Flat-Walker-v0 \
    --motion_file source/motion/walker/npz/walk_forward.npz \
    --num_envs    16 \
    --load_run    2024-01-01_12-00-00_walker_flat \
    --load_checkpoint model_10000.pt
```

### 4.2 回放 NPZ 运动参考

仅查看参考运动（不加载策略）：

```bash
python scripts/replay_npz.py \
    --motion_file source/motion/walker/npz/walk_forward.npz \
    --robot walker
```

---

## 5. 多运动基础模型训练（Phase 2/3）

多运动训练使用 `MultiMotionCommand`，单个策略网络学习同时执行多种参考运动，通过运动 ID 编码区分不同运动。

### 5.1 快速开始

```bash
python scripts/rsl_rl/train.py \
    --task          Tracking-Flat-Walker-v0 \
    --motion_files  source/motion/walker/npz/walk.npz \
                    source/motion/walker/npz/run.npz \
                    source/motion/walker/npz/jump.npz \
    --motion_selector uniform \
    --num_envs      4096 \
    --headless
```

> **说明**：使用 `--motion_files`（复数）替代 `--motion_file`（单数）即可启用多运动模式。
> 该参数需要目标任务配置使用 `MultiMotionCommandCfg`。

### 5.2 运动选择策略

通过 `--motion_selector` 选择运动分配策略：

| 策略 | 参数值 | 说明 |
|------|--------|------|
| 均匀随机 | `uniform` | 每次重置时均匀随机分配运动（默认，适合探索阶段） |
| 自适应 | `adaptive` | 优先训练成功率低的运动（聚焦于困难运动） |
| 课程 | `curriculum` | 从第一个运动开始，成功率超过阈值后依次解锁后续运动 |

```bash
# 课程学习：先掌握 walk，再解锁 run，最后解锁 jump
python scripts/rsl_rl/train.py \
    --task Tracking-Flat-Walker-v0 \
    --motion_files walk.npz run.npz jump.npz \
    --motion_selector curriculum \
    --num_envs 4096 --headless

# 自适应：专注于当前最困难的运动
python scripts/rsl_rl/train.py \
    --task Tracking-Flat-Walker-v0 \
    --motion_files walk.npz run.npz jump.npz \
    --motion_selector adaptive \
    --num_envs 4096 --headless
```

### 5.3 多运动观测空间

策略观测向量在单运动基础上新增三个分量：

| 观测项 | 维度 | 说明 |
|--------|------|------|
| `motion_id_encoding` | `num_motions` | 当前运动的 one-hot 编码（<50 个运动时） |
| `motion_id_embedding` | `embedding_dim`（默认 16） | 连续 embedding（≥50 个运动时自动启用） |
| `motion_change_signal` | 1 | 运动切换后 5 步内为 1，否则为 0 |
| `motion_progress` | 1 | 当前运动进度（0.0 → 1.0） |

**观测总维度**（<50 个运动）：

```
原始维度 + num_motions + 2
```

### 5.4 运动数量与 Embedding 模式

| 运动数量 | 编码方式 | 观测新增维度 |
|---------|---------|-------------|
| 1–49 | one-hot | = `num_motions` |
| ≥50（自动）| embedding（dim=16） | 16（固定） |
| 任意（手动） | 由 `use_embedding` 控制 | `embedding_dim` |

在环境配置中手动设置 embedding：

```python
cfg.commands.motion.use_embedding = True   # 强制启用
cfg.commands.motion.embedding_dim = 32     # 调大 embedding 维度
```

### 5.5 课程学习建议流程

```
Stage 1 (0–5k iter):    单运动基线，掌握最简单的运动
Stage 2 (5–15k iter):   加入第 2 个运动（curriculum 自动解锁）
Stage 3 (15–30k iter):  加入第 3–5 个运动
Stage 4 (30–50k iter):  全库 adaptive 采样，强化鲁棒性
```

---

## 6. 配置参考

### 6.1 PPO 超参数（`rsl_rl_ppo_cfg.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_iterations` | 50000 | 总训练迭代数 |
| `num_steps_per_env` | 24 | 每次 rollout 的步数 |
| `num_learning_epochs` | 5 | 每次 rollout 的 PPO 优化轮数 |
| `num_mini_batches` | 4 | mini-batch 数量 |
| `learning_rate` | 1e-3 | 初始学习率（adaptive schedule） |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ |
| `clip_param` | 0.2 | PPO clip 系数 |
| `entropy_coef` | 0.005 | 熵正则系数 |
| `desired_kl` | 0.01 | 自适应 lr 的目标 KL 散度 |
| Actor/Critic 网络 | [512, 256, 128] | 三层 MLP，ELU 激活 |

### 6.2 奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `motion_global_anchor_pos` | +0.5 | 锚点（骨盆）全局位置误差 |
| `motion_global_anchor_ori` | +0.5 | 锚点全局朝向误差 |
| `motion_body_pos` | +1.0 | 全身相对位置误差 |
| `motion_body_ori` | +1.0 | 全身相对朝向误差 |
| `motion_body_lin_vel` | +1.0 | 全身线速度误差 |
| `motion_body_ang_vel` | +1.0 | 全身角速度误差 |
| `action_rate_l2` | −0.1 | 动作变化率惩罚（平滑性） |
| `joint_limit` | −10.0 | 关节限位惩罚 |
| `undesired_contacts` | −0.1 | 非预期接触惩罚（脚踝/手腕除外） |

所有追踪误差项使用 `exp(-error² / std²)` 形式，`std` 越小要求精度越高。

### 6.3 `MultiMotionCommandCfg` 关键字段

```python
from whole_body_tracking.tasks.tracking.mdp.commands import MultiMotionCommandCfg

cfg = MultiMotionCommandCfg(
    asset_name        = "robot",
    motion_library_dir= "",                  # 元数据目录（可选）
    motion_files      = ["a.npz", "b.npz"],  # 运动文件列表
    anchor_body_name  = "pelvis",
    body_names        = ["pelvis", "knee_pitch_l_link", ...],

    # 自适应采样
    adaptive_kernel_size  = 1,
    adaptive_lambda       = 0.8,
    adaptive_uniform_ratio= 0.1,    # 均匀采样占比（防止 bin 饥饿）
    adaptive_alpha        = 0.001,  # 失败率 EMA 衰减

    # 运动选择器
    motion_selector_type          = "curriculum",  # uniform / adaptive / curriculum
    curriculum_success_threshold  = 0.7,           # 解锁下一运动的成功率阈值
    curriculum_window             = 1000,           # 滑动窗口大小（episodes）

    # Embedding（≥50 运动时自动启用）
    use_embedding = None,   # None=自动, True=强制, False=禁用
    embedding_dim = 16,
)
```

---

## 7. ONNX 导出与部署

### 7.1 自动导出

训练使用 WandB logger 时，每次保存检查点都会自动导出 ONNX 文件：

```bash
python scripts/rsl_rl/train.py \
    --task Tracking-Flat-Walker-v0 \
    --motion_file walk.npz \
    --logger wandb --log_project_name my_project \
    --headless
```

### 7.2 手动导出（play 时导出）

```bash
python scripts/rsl_rl/play.py \
    --task        Tracking-Flat-Walker-v0 \
    --motion_file walk.npz \
    --load_run    <run_name> \
    --num_envs    1
```

play 脚本在加载检查点后会自动导出到 `logs/.../exported/` 目录。

### 7.3 多运动 ONNX 模型

多运动策略导出的 ONNX 模型有三个输入：

| 输入 | 形状 | 说明 |
|------|------|------|
| `obs` | `[1, obs_dim]` | 策略观测向量 |
| `motion_id` | `[1, 1]` | 当前运动 ID（整数） |
| `time_step` | `[1, 1]` | 当前帧编号（整数） |

七个输出（与单运动版本相同）：

| 输出 | 说明 |
|------|------|
| `actions` | 关节位置动作 |
| `joint_pos` | 参考关节角度 |
| `joint_vel` | 参考关节速度 |
| `body_pos_w` | 参考身体位置（世界系） |
| `body_quat_w` | 参考身体姿态（世界系） |
| `body_lin_vel_w` | 参考线速度（世界系） |
| `body_ang_vel_w` | 参考角速度（世界系） |

### 7.4 读取 ONNX 元数据

```python
import onnx

model = onnx.load("policy.onnx")
meta = {p.key: p.value for p in model.metadata_props}

print(meta["joint_names"])         # 逗号分隔的关节名
print(meta["anchor_body_name"])    # 锚点身体名
print(meta["body_names"])          # 追踪身体名列表
print(meta["action_scale"])        # 动作缩放系数

# 多运动模型额外字段
print(meta["num_motions"])         # 运动数量
print(meta["motion_names"])        # 各运动名称
print(meta["motion_selector_type"])# 训练时使用的选择策略
```

---

## 8. 性能分析

```bash
# 默认配置（CPU，4096 envs，1000 帧/运动）
conda run -n env_isaaclab python scripts/profile_multi_motion.py

# 自定义配置
conda run -n env_isaaclab python scripts/profile_multi_motion.py \
    --device cuda \
    --num_envs 4096 \
    --num_frames 1000 \
    --num_motions_list 1 5 10 20 50 100
```

**实测参考数据**（CPU，4096 envs，800 帧/运动）：

| 指标 | 值 |
|------|-----|
| 堆叠张量查找开销（vs 单运动）| 0–80%（绝对 <25 µs） |
| Embedding 查找延迟 | ~15–25 µs（与运动数量无关） |
| Curriculum 选择延迟 | ~9–13 µs |
| Adaptive 选择延迟 | 150–700 µs（50 运动时） |
| 内存占用（100 运动 × 800 帧）| ~65 MB |

---

## 9. 扩展：添加新机器人或新任务

### 9.1 添加新机器人

1. 在 `source/.../robots/` 创建机器人配置文件（参考 `walker.py`）
2. 在 `source/.../tasks/tracking/config/<robot>/` 创建任务配置：
   - `<robot>_tracking_env_cfg.py` — 继承 `TrackingEnvCfg`，设置奖励/终止条件
   - `flat_env_cfg.py` — 设置场景、身体追踪列表、锚点
   - `agents/rsl_rl_ppo_cfg.py` — PPO 超参数
   - `__init__.py` — 注册 gym 任务

### 9.2 添加新运动

只需准备 `.npz` 文件（见[第 2 节](#2-运动数据准备)），无需修改代码。

**多运动训练时指定所有运动：**

```bash
python scripts/rsl_rl/train.py \
    --task Tracking-Flat-Walker-v0 \
    --motion_files motion1.npz motion2.npz motion3.npz \
    --motion_selector curriculum
```

### 9.3 自定义运动选择器

继承 `MotionSelector` 并实现两个方法：

```python
from whole_body_tracking.tasks.tracking.mdp.motion_selector import MotionSelector
import torch

class MySelector(MotionSelector):
    def select_motions(self, env_ids, command) -> torch.Tensor:
        # 返回 LongTensor [len(env_ids)]，每个元素为下一运动 ID
        ...

    def update(self, motion_ids: torch.Tensor, success_flags: torch.Tensor) -> None:
        # 记录 episode 结果以更新选择策略
        ...
```

然后在 `MultiMotionCommand` 创建后挂载：

```python
command = env.command_manager.get_term("motion")
command.motion_selector = MySelector()
```

---

## 常见问题

**Q: 训练时 GPU 显存不足**
A: 减少 `--num_envs`，或减少 `motion_files` 中的运动数量。50 个运动 × 1000 帧 ≈ 40 MB。

**Q: 多运动训练时策略混乱（无法跟踪任何运动）**
A: 使用 `--motion_selector curriculum`，从最简单的运动开始逐步扩展。避免一开始就引入差异过大的运动。

**Q: `--motion_files` 无效，仍然使用单运动模式**
A: 确认 `--task` 对应的 env 配置使用了 `MultiMotionCommandCfg`（非 `MotionCommandCfg`）。

**Q: NPZ 文件报 `KeyError`**
A: 检查 `csv_to_npz.py` 使用的 `--robot` 参数是否与实际机器人匹配。不同机器人的关节顺序不同。

**Q: play 时导出 ONNX 失败**
A: 检查 `onnx` 包是否安装：`pip install onnx`。
