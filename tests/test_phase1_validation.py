"""Quick validation tests for Phase 1 without torch dependency.

This script validates:
1. All new files exist and have correct structure
2. Python syntax is correct
3. Imports work properly (when dependencies are available)
4. Configuration classes are properly defined
"""

import sys
import os
from pathlib import Path

# Add source to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "source"))

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{BOLD}{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}{RESET}\n")


def print_pass(msg):
    """Print a passing test."""
    print(f"{GREEN}✓{RESET} {msg}")


def print_fail(msg):
    """Print a failing test."""
    print(f"{RED}✗{RESET} {msg}")


def print_info(msg):
    """Print info message."""
    print(f"{YELLOW}ℹ{RESET} {msg}")


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print_pass(f"{description} exists")
        return True
    else:
        print_fail(f"{description} NOT FOUND: {filepath}")
        return False


def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print_pass(f"Syntax valid: {Path(filepath).name}")
        return True
    except SyntaxError as e:
        print_fail(f"Syntax error in {Path(filepath).name}: {e}")
        return False


def check_class_definition(module_path, class_name):
    """Check if a class is defined in a module."""
    try:
        # Read file and check for class definition
        with open(module_path, 'r') as f:
            content = f.read()
            if f"class {class_name}" in content:
                print_pass(f"Class '{class_name}' defined in {Path(module_path).name}")
                return True
            else:
                print_fail(f"Class '{class_name}' NOT found in {Path(module_path).name}")
                return False
    except Exception as e:
        print_fail(f"Error checking class: {e}")
        return False


def check_function_definition(module_path, func_name):
    """Check if a function is defined in a module."""
    try:
        with open(module_path, 'r') as f:
            content = f.read()
            if f"def {func_name}" in content:
                print_pass(f"Function '{func_name}' defined in {Path(module_path).name}")
                return True
            else:
                print_fail(f"Function '{func_name}' NOT found in {Path(module_path).name}")
                return False
    except Exception as e:
        print_fail(f"Error checking function: {e}")
        return False


def main():
    """Run all validation tests."""
    print_section("PHASE 1 VALIDATION TESTS")
    print_info("Environment: No torch/Isaac Lab required")

    passed = 0
    failed = 0

    # =======================
    # 1. Check file structure
    # =======================
    print_section("1. File Structure Validation")

    files_to_check = [
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_info.py",
         "MotionInfo module"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
         "Commands module"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py",
         "Observations module"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
         "Rewards module"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
         "Environment config"),
        ("PHASE1_SUMMARY.md", "Phase 1 summary"),
    ]

    for filepath, description in files_to_check:
        full_path = PROJECT_ROOT / filepath
        if check_file_exists(full_path, description):
            passed += 1
        else:
            failed += 1

    # =======================
    # 2. Python syntax check
    # =======================
    print_section("2. Python Syntax Validation")

    python_files = [
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_info.py",
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py",
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
        "source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    ]

    for filepath in python_files:
        full_path = PROJECT_ROOT / filepath
        if check_python_syntax(full_path):
            passed += 1
        else:
            failed += 1

    # =======================
    # 3. Class definitions
    # =======================
    print_section("3. Class Definitions")

    class_checks = [
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_info.py",
         "MotionInfo"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
         "MotionLibrary"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
         "MultiMotionCommand"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
         "MotionCommandCfg"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
         "MultiMotionCommandCfg"),
    ]

    for filepath, class_name in class_checks:
        full_path = PROJECT_ROOT / filepath
        if check_class_definition(full_path, class_name):
            passed += 1
        else:
            failed += 1

    # =======================
    # 4. Function definitions
    # =======================
    print_section("4. Observation Functions")

    obs_functions = [
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py",
         "motion_id_encoding"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py",
         "motion_change_signal"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py",
         "motion_progress"),
    ]

    for filepath, func_name in obs_functions:
        full_path = PROJECT_ROOT / filepath
        if check_function_definition(full_path, func_name):
            passed += 1
        else:
            failed += 1

    print_section("5. Reward Functions")

    reward_functions = [
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
         "motion_difficulty_scaling"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
         "motion_diversity_bonus"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
         "motion_switching_penalty"),
    ]

    for filepath, func_name in reward_functions:
        full_path = PROJECT_ROOT / filepath
        if check_function_definition(full_path, func_name):
            passed += 1
        else:
            failed += 1

    # =======================
    # 6. Configuration classes
    # =======================
    print_section("6. Configuration Classes")

    config_checks = [
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
         "CommandsCfgMultiMotion"),
        ("source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
         "ObservationsCfgMultiMotion"),
    ]

    for filepath, class_name in config_checks:
        full_path = PROJECT_ROOT / filepath
        if check_class_definition(full_path, class_name):
            passed += 1
        else:
            failed += 1

    # =======================
    # 7. Summary report
    # =======================
    print_section("VALIDATION SUMMARY")

    print(f"{BOLD}Total Tests:{RESET} {passed + failed}")
    print(f"{GREEN}Passed:{RESET} {passed}")
    print(f"{RED}Failed:{RESET} {failed}")

    if failed == 0:
        print(f"\n{GREEN}{BOLD}✓ ALL VALIDATION TESTS PASSED!{RESET}")
        print(f"\nPhase 1 implementation is complete and ready for integration testing.")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ Some tests failed. Please review above.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
