"""Performance profiling for multi-motion training components.

Measures:
  1. Memory footprint   – stacked tensors vs individual MotionLoader objects
  2. Per-step latency   – stacked tensor lookup (per-env indexing) throughput
  3. Embedding latency  – MotionEmbeddingBank.get() throughput
  4. Selector overhead  – motion selection latency per strategy
  5. ONNX model size    – exported file size vs num_motions

Run:
    conda run -n env_isaaclab python scripts/profile_multi_motion.py
    conda run -n env_isaaclab python scripts/profile_multi_motion.py --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import time
import sys
import os

import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Profile multi-motion components")
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--num_motions_list", type=int, nargs="+",
                    default=[1, 5, 10, 20, 50, 100])
parser.add_argument("--num_frames", type=int, default=1000,
                    help="Frames per motion (uniform)")
parser.add_argument("--num_joints", type=int, default=18)
parser.add_argument("--num_bodies", type=int, default=13)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--steps", type=int, default=500)
args = parser.parse_args()

DEV = torch.device(args.device)
NE  = args.num_envs
NF  = args.num_frames
NJ  = args.num_joints
NB  = args.num_bodies
SEP = "─" * 70


def fmt_mb(b: int) -> str: return f"{b / 1e6:.1f} MB"
def fmt_us(s: float) -> str: return f"{s * 1e6:.1f} µs"
def fmt_ms(s: float) -> str: return f"{s * 1e3:.2f} ms"


def timed(func, warmup=None, steps=None) -> float:
    warmup = warmup or args.warmup
    steps  = steps  or args.steps
    for _ in range(warmup):
        func()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        func()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def _direct_import(module_name: str, file_path: str):
    """Import a Python file without triggering its package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPO = os.path.join(os.path.dirname(__file__), "..")
MDPdir  = os.path.join(REPO, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp")
UTILdir = os.path.join(REPO, "source/whole_body_tracking/whole_body_tracking/utils")


# ---------------------------------------------------------------------------
# 1. Memory footprint
# ---------------------------------------------------------------------------

def section_memory():
    print(f"\n{SEP}")
    print(f"1. MEMORY FOOTPRINT  (num_frames={NF}, joints={NJ}, bodies={NB})")
    print(SEP)
    print(f"{'Num motions':>12}  {'Stacked tensors':>18}  {'Per-motion dicts':>18}")
    print(SEP)

    bytes_per_frame = (
        NJ * 4 * 2        # joint_pos + joint_vel
        + NB * 3 * 4 * 2  # body_pos_w + body_lin_vel_w
        + NB * 4 * 4      # body_quat_w
        + NB * 3 * 4      # body_ang_vel_w
    )

    for nm in args.num_motions_list:
        stacked = nm * NF * bytes_per_frame
        per_mot = nm * (NF * bytes_per_frame + 128)  # +128 per-dict Python overhead
        print(f"{nm:>12}  {fmt_mb(stacked):>18}  {fmt_mb(per_mot):>18}")


# ---------------------------------------------------------------------------
# 2. Per-step stacked tensor lookup
# ---------------------------------------------------------------------------

def section_tensor_lookup():
    print(f"\n{SEP}")
    print(f"2. PER-STEP TENSOR LOOKUP LATENCY  (num_envs={NE})")
    print(SEP)
    print(f"{'Num motions':>12}  {'Stacked [M, T, ...]':>22}  {'Single motion [T, ...]':>24}")
    print(SEP)

    for nm in args.num_motions_list:
        all_jp    = torch.randn(nm, NF, NJ, device=DEV)
        motion_ids = torch.randint(0, nm, (NE,), device=DEV)
        time_steps = torch.randint(0, NF, (NE,), device=DEV)

        def stacked():
            return all_jp[motion_ids, time_steps]

        single_jp = torch.randn(NF, NJ, device=DEV)
        def single():
            return single_jp[time_steps]

        ts = timed(stacked)
        tsi = timed(single)
        overhead = (ts / tsi - 1.0) * 100
        print(f"{nm:>12}  {fmt_us(ts):>22}  {fmt_us(tsi):>24}  (overhead: {overhead:+.0f}%)")


# ---------------------------------------------------------------------------
# 3. Embedding bank
# ---------------------------------------------------------------------------

def section_embedding():
    print(f"\n{SEP}")
    print(f"3. MOTION EMBEDDING BANK  (num_envs={NE})")
    print(SEP)

    try:
        emb_mod = _direct_import(
            "motion_embedding",
            os.path.join(UTILdir, "motion_embedding.py"),
        )
        MotionEmbeddingBank = emb_mod.MotionEmbeddingBank
    except Exception as e:
        print(f"  [skip] {e}")
        return

    print(f"{'Num motions':>12}  {'embed_dim':>10}  {'Latency / step':>18}  {'Table (MB)':>10}")
    print(SEP)

    for nm in args.num_motions_list:
        for edim in [16, 32]:
            bank = MotionEmbeddingBank(nm, edim, device=DEV)
            mids = torch.randint(0, nm, (NE,), device=DEV)

            def emb_lookup(): return bank.get(mids)

            t = timed(emb_lookup)
            table_bytes = nm * edim * 4
            print(f"{nm:>12}  {edim:>10}  {fmt_us(t):>18}  {fmt_mb(table_bytes):>10}")


# ---------------------------------------------------------------------------
# 4. Selector overhead
# ---------------------------------------------------------------------------

def section_selector():
    print(f"\n{SEP}")
    print(f"4. MOTION SELECTOR LATENCY  (n_reset ≈ num_envs × 1%)")
    print(SEP)

    try:
        sel_mod = _direct_import(
            "motion_selector",
            os.path.join(MDPdir, "motion_selector.py"),
        )
        UniformMotionSelector    = sel_mod.UniformMotionSelector
        AdaptiveMotionSelector   = sel_mod.AdaptiveMotionSelector
        CurriculumMotionSelector = sel_mod.CurriculumMotionSelector
    except Exception as e:
        print(f"  [skip] {e}")
        return

    class FakeCmd:
        def __init__(self, nm):
            self.motion_library = type("L", (), {"num_motions": nm})()
            self.device = DEV

    n_reset   = max(1, NE // 100)
    env_ids   = torch.arange(n_reset, device=DEV)

    print(f"{'Selector':>22}  {'Num motions':>12}  {'Latency / call':>18}")
    print(SEP)

    for nm in [5, 20, 50]:
        fake    = FakeCmd(nm)
        mids    = torch.randint(0, nm, (n_reset,), device=DEV)
        success = torch.rand(n_reset, device=DEV) > 0.3

        pairs = [
            ("Uniform",    UniformMotionSelector()),
            ("Adaptive",   AdaptiveMotionSelector()),
            ("Curriculum", CurriculumMotionSelector(nm)),
        ]
        for name, sel in pairs:
            sel.update(mids, success)  # warm-up state

            def sel_call(s=sel):
                s.update(mids, success)
                return s.select_motions(env_ids, fake)

            t = timed(sel_call)
            print(f"{name:>22}  {nm:>12}  {fmt_us(t):>18}")


# ---------------------------------------------------------------------------
# 5. ONNX model size estimate
# ---------------------------------------------------------------------------

def section_onnx_size():
    print(f"\n{SEP}")
    print(f"5. ONNX MODEL SIZE ESTIMATE  (joints={NJ}, bodies={NB})")
    print(SEP)
    print("  (Motion data tensors only; excludes policy MLP weights)")
    print(f"{'Num motions':>12}  {'Frames':>10}  {'Data tensors':>16}  {'vs single-motion':>18}")
    print(SEP)

    bpf = NJ * 4 * 2 + NB * 3 * 4 * 2 + NB * 4 * 4 + NB * 3 * 4
    baseline = NF * bpf

    for nm in args.num_motions_list:
        total = nm * NF * bpf
        print(f"{nm:>12}  {NF:>10}  {fmt_mb(total):>16}  {total/baseline:>17.1f}×")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  MULTI-MOTION PERFORMANCE PROFILE")
    print(f"  device={args.device}  num_envs={NE}  frames/motion={NF}")
    print(f"{'='*70}")

    section_memory()
    section_tensor_lookup()
    section_embedding()
    section_selector()
    section_onnx_size()

    print(f"\n{'='*70}\n  Done.\n{'='*70}\n")
