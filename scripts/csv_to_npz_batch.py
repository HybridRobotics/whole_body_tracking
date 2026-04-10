import os
import argparse
import subprocess
import sys
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="Batch convert CSV files to NPZ using csv_to_npz.py")

    # 必需参数
    parser.add_argument("--input_dir", type=str, required=True, help="包含CSV文件的文件夹路径（递归搜索子目录）")
    parser.add_argument("--output_dir", type=str, required=True, help="保存NPZ文件的文件夹路径（保留子目录结构）")
    parser.add_argument("--robot", type=str, required=True, help="机器人名称 (例如: tienkung, unitree_g1, walker)")

    # 可选参数
    parser.add_argument("--input_fps", type=int, default=30, help="输入数据的FPS (默认: 30)")
    parser.add_argument("--script_path", type=str, default="scripts/csv_to_npz.py", help="单文件转换脚本的路径")
    parser.add_argument("--no_wandb", action="store_true", help="跳过 WandB 上传，仅本地保存 NPZ")

    return parser.parse_args()


def main():
    args = get_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # 1. 检查输入目录是否存在
    if not input_dir.exists():
        print(f"[Error] 输入目录不存在: {input_dir}")
        return

    # 2. 递归搜索所有 CSV 文件
    csv_files = sorted(input_dir.rglob("*.csv"))
    total_files = len(csv_files)

    if total_files == 0:
        print(f"[Warning] 在 {input_dir} 中（含子目录）没有找到 .csv 文件")
        return

    print(f"======== 开始批量处理 {total_files} 个文件 ========")

    success_count = 0
    fail_count = 0

    # 3. 循环处理
    for i, input_path in enumerate(csv_files):
        # 保留相对于 input_dir 的子目录结构
        rel_path = input_path.relative_to(input_dir)
        output_name = output_dir / rel_path.with_suffix("")

        # 确保输出子目录存在
        output_name.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[{i + 1}/{total_files}] 正在处理: {rel_path} ...")

        # 构建调用命令，使用 sys.executable 确保相同 Python 环境
        cmd = [
            sys.executable, args.script_path,
            "--input_file", str(input_path),
            "--input_fps", str(args.input_fps),
            "--output_name", str(output_name),
            "--robot", args.robot,
            "--headless",  # suppress Isaac Sim GUI for batch processing
        ]
        if args.no_wandb:
            cmd.append("--no_wandb")

        try:
            subprocess.run(cmd, check=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[Error] 处理 {rel_path} 失败! 错误代码: {e.returncode}")
            fail_count += 1
        except Exception as e:
            print(f"[Error] 发生未知错误: {e}")
            fail_count += 1

    print(f"\n======== 批量处理完成! 成功: {success_count}, 失败: {fail_count} ========")
    print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()