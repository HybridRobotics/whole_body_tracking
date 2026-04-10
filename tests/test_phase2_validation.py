"""Phase 2 validation tests — no torch/Isaac Lab required.

Checks:
  1. motion_selector.py exists and has valid syntax
  2. All three selector classes are defined
  3. MotionLibrary.build_stacked_tensors is defined
  4. MultiMotionCommand uses per-env stacked tensors (no _get_current_motion)
  5. MultiMotionCommandCfg has selector config fields
  6. train.py has --motion_files argument
"""
import ast
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..")
MDPdir = os.path.join(BASE, "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp")

RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def _banner(title: str):
    print(f"\n{BOLD}{'='*70}\n{title}\n{'='*70}{RESET}")


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def check_file_exists(path: str, label: str) -> bool:
    ok = os.path.isfile(path)
    print(f"{'✓' if ok else '✗'} {label}")
    return ok


def check_syntax(path: str, label: str) -> tuple[bool, ast.Module | None]:
    try:
        src = _read(path)
        tree = ast.parse(src)
        print(f"{GREEN}✓{RESET} Syntax valid: {label}")
        return True, tree
    except SyntaxError as e:
        print(f"{RED}✗{RESET} Syntax error in {label}: line {e.lineno}: {e.msg}")
        return False, None


def check_class_in_source(src: str, classname: str, file_label: str) -> bool:
    ok = f"class {classname}" in src
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"{mark} Class '{classname}' in {file_label}")
    return ok


def check_method_in_source(src: str, method: str, file_label: str) -> bool:
    ok = f"def {method}" in src
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"{mark} Method/function '{method}' in {file_label}")
    return ok


def check_string_in_source(src: str, needle: str, description: str) -> bool:
    ok = needle in src
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"{mark} {description}")
    return ok


def check_string_absent(src: str, needle: str, description: str) -> bool:
    ok = needle not in src
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"{mark} {description}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*70}\nPHASE 2 VALIDATION TESTS\n{'='*70}{RESET}")
    print(f"\033[93mℹ{RESET} Environment: No torch/Isaac Lab required\n")

    results: list[bool] = []

    # ── 1. File structure ────────────────────────────────────────────────────
    _banner("1. File Structure")
    selector_path  = os.path.join(MDPdir, "motion_selector.py")
    commands_path  = os.path.join(MDPdir, "commands.py")
    train_path     = os.path.join(BASE, "scripts/rsl_rl/train.py")

    results.append(check_file_exists(selector_path, "motion_selector.py exists"))
    results.append(check_file_exists(commands_path, "commands.py exists"))

    # ── 2. Syntax ────────────────────────────────────────────────────────────
    _banner("2. Python Syntax")
    ok_sel, _ = check_syntax(selector_path, "motion_selector.py")
    ok_cmd, _ = check_syntax(commands_path, "commands.py")
    results += [ok_sel, ok_cmd]

    sel_src = _read(selector_path) if ok_sel else ""
    cmd_src = _read(commands_path) if ok_cmd else ""
    train_src = _read(train_path) if os.path.isfile(train_path) else ""

    # ── 3. Selector classes ──────────────────────────────────────────────────
    _banner("3. MotionSelector Classes")
    for cls in ["MotionSelector", "UniformMotionSelector",
                "AdaptiveMotionSelector", "CurriculumMotionSelector"]:
        results.append(check_class_in_source(sel_src, cls, "motion_selector.py"))

    # ── 4. Selector methods ──────────────────────────────────────────────────
    _banner("4. Selector Interface Methods")
    for fn in ["select_motions", "update"]:
        results.append(check_method_in_source(sel_src, fn, "motion_selector.py"))

    # ── 5. MotionLibrary.build_stacked_tensors ───────────────────────────────
    _banner("5. MotionLibrary Stacked Tensor Support")
    results.append(check_method_in_source(cmd_src, "build_stacked_tensors", "commands.py"))
    results.append(check_string_in_source(cmd_src, "_all_joint_pos", "MultiMotionCommand uses _all_joint_pos"))
    results.append(check_string_in_source(cmd_src, "motion_time_totals", "MultiMotionCommand has motion_time_totals"))
    results.append(check_string_in_source(cmd_src, "_clamped_time_steps", "MultiMotionCommand has _clamped_time_steps"))

    # ── 6. Per-env fix: no _get_current_motion ───────────────────────────────
    _banner("6. Per-env Motion Bug Fixed")
    results.append(check_string_absent(cmd_src, "_get_current_motion", "No _get_current_motion (broken method removed)"))
    results.append(check_string_in_source(cmd_src, "_select_motions", "MultiMotionCommand has _select_motions"))
    results.append(check_string_in_source(cmd_src, "_init_motion_selector", "MultiMotionCommand has _init_motion_selector"))

    # ── 7. MultiMotionCommandCfg selector fields ─────────────────────────────
    _banner("7. MultiMotionCommandCfg Selector Config Fields")
    results.append(check_string_in_source(cmd_src, "motion_selector_type", "motion_selector_type field"))
    results.append(check_string_in_source(cmd_src, "curriculum_success_threshold", "curriculum_success_threshold field"))
    results.append(check_string_in_source(cmd_src, "curriculum_window", "curriculum_window field"))

    # ── 8. train.py multi-motion support ────────────────────────────────────
    _banner("8. train.py Multi-motion Support")
    results.append(check_string_in_source(train_src, "--motion_files", "train.py has --motion_files argument"))
    results.append(check_string_in_source(train_src, "--motion_selector", "train.py has --motion_selector argument"))
    results.append(check_string_in_source(train_src, "MultiMotionCommandCfg", "train.py imports MultiMotionCommandCfg"))

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("VALIDATION SUMMARY")
    passed = sum(results)
    total  = len(results)
    print(f"{BOLD}Total Tests:{RESET} {total}")
    print(f"{GREEN}Passed:{RESET} {passed}")
    print(f"{RED}Failed:{RESET} {total - passed}")
    if passed == total:
        print(f"\n{GREEN}{BOLD}✓ ALL PHASE 2 VALIDATION TESTS PASSED!{RESET}\n")
    else:
        failed = [(i+1) for i, r in enumerate(results) if not r]
        print(f"\n{RED}{BOLD}✗ {total - passed} tests failed (indices: {failed}){RESET}\n")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
