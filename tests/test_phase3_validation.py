"""Phase 3 validation tests — no torch/Isaac Lab required.

Checks:
  1. motion_embedding.py exists with valid syntax
  2. MotionEmbeddingBank class and methods defined
  3. motion_id_embedding observation function added
  4. motion_progress uses vectorised path (no Python loop)
  5. ONNX exporter has multi-motion support
  6. attach_onnx_metadata has multi-motion metadata keys
  7. Profiling script exists and has valid syntax
  8. MultiMotionCommandCfg has embedding config fields
"""

import ast
import os
import sys

BASE   = os.path.join(os.path.dirname(__file__), "..")
MDPdir = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp")
UTILS  = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/utils")

GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"


def _banner(title): print(f"\n{BOLD}{'='*70}\n{title}\n{'='*70}{RESET}")
def _read(p): return open(p).read()
def ok(msg): print(f"{GREEN}✓{RESET} {msg}"); return True
def fail(msg): print(f"{RED}✗{RESET} {msg}"); return False


def check_file(path, label):
    if os.path.isfile(path): return ok(label)
    return fail(label)


def check_syntax(path, label):
    try:
        ast.parse(_read(path)); return ok(f"Syntax valid: {label}")
    except SyntaxError as e:
        return fail(f"Syntax error in {label}: line {e.lineno}: {e.msg}")


def check_in(src, needle, desc):
    if needle in src: return ok(desc)
    return fail(desc)


def check_absent(src, needle, desc):
    if needle not in src: return ok(desc)
    return fail(desc)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*70}\nPHASE 3 VALIDATION TESTS\n{'='*70}{RESET}")
    R = []

    # ── 1. File structure ────────────────────────────────────────────────────
    _banner("1. File Structure")
    emb_path  = os.path.join(UTILS, "motion_embedding.py")
    exp_path  = os.path.join(UTILS, "exporter.py")
    obs_path  = os.path.join(MDPdir, "observations.py")
    cmd_path  = os.path.join(MDPdir, "commands.py")
    prof_path = os.path.join(BASE, "scripts/profile_multi_motion.py")

    R.append(check_file(emb_path,  "motion_embedding.py exists"))
    R.append(check_file(exp_path,  "exporter.py exists"))
    R.append(check_file(prof_path, "profile_multi_motion.py exists"))

    # ── 2. Syntax ────────────────────────────────────────────────────────────
    _banner("2. Python Syntax")
    R.append(check_syntax(emb_path, "motion_embedding.py"))
    R.append(check_syntax(exp_path, "exporter.py"))
    R.append(check_syntax(prof_path, "profile_multi_motion.py"))

    emb_src  = _read(emb_path)
    exp_src  = _read(exp_path)
    obs_src  = _read(obs_path)
    cmd_src  = _read(cmd_path)

    # ── 3. MotionEmbeddingBank ───────────────────────────────────────────────
    _banner("3. MotionEmbeddingBank Class")
    R.append(check_in(emb_src, "class MotionEmbeddingBank", "MotionEmbeddingBank defined"))
    R.append(check_in(emb_src, "def get(",         "get() method"))
    R.append(check_in(emb_src, "def update_from_motion_info(", "update_from_motion_info() method"))
    R.append(check_in(emb_src, "def state_dict(",  "state_dict() method"))
    R.append(check_in(emb_src, "def load_state_dict(", "load_state_dict() method"))

    # ── 4. Observation functions ─────────────────────────────────────────────
    _banner("4. Observation Functions")
    R.append(check_in(obs_src, "def motion_id_embedding(", "motion_id_embedding() added"))
    R.append(check_in(obs_src, "embedding_bank", "motion_id_embedding uses embedding_bank"))
    R.append(check_absent(obs_src, "for env_id, motion_id in enumerate", "motion_progress vectorised (no Python loop)"))
    R.append(check_in(obs_src, "motion_time_totals", "motion_progress uses motion_time_totals"))

    # ── 5. MultiMotionCommand embedding integration ──────────────────────────
    _banner("5. MultiMotionCommand Embedding Integration")
    R.append(check_in(cmd_src, "_init_embedding_bank", "_init_embedding_bank method"))
    R.append(check_in(cmd_src, "embedding_bank", "embedding_bank attribute initialized"))
    R.append(check_in(cmd_src, "use_embedding",  "use_embedding config field"))
    R.append(check_in(cmd_src, "embedding_dim",  "embedding_dim config field"))

    # ── 6. ONNX multi-motion exporter ────────────────────────────────────────
    _banner("6. ONNX Multi-Motion Exporter")
    R.append(check_in(exp_src, "class _OnnxMultiMotionPolicyExporter", "multi-motion exporter class"))
    R.append(check_in(exp_src, "def export_multi_motion_policy_as_onnx", "export function"))
    R.append(check_in(exp_src, "motion_id", "motion_id input in forward()"))
    R.append(check_in(exp_src, '"motion_id"', "motion_id ONNX input name"))
    R.append(check_in(exp_src, "num_motions", "num_motions metadata"))
    R.append(check_in(exp_src, "motion_names", "motion_names metadata"))
    R.append(check_in(exp_src, "motion_selector_type", "motion_selector_type metadata"))
    R.append(check_in(exp_src, "isinstance(cmd, MultiMotionCommand)", "auto-dispatch in export_motion_policy_as_onnx"))

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("VALIDATION SUMMARY")
    passed, total = sum(R), len(R)
    print(f"{BOLD}Total Tests:{RESET} {total}")
    print(f"{GREEN}Passed:{RESET} {passed}")
    print(f"{RED}Failed:{RESET} {total - passed}")
    if passed == total:
        print(f"\n{GREEN}{BOLD}✓ ALL PHASE 3 VALIDATION TESTS PASSED!{RESET}\n")
    else:
        failed = [i+1 for i,r in enumerate(R) if not r]
        print(f"\n{RED}{BOLD}✗ {total-passed} failed (indices: {failed}){RESET}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
