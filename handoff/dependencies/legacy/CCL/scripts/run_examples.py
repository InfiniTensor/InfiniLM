#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ==============================================================================
# DISCOVERY UTILITIES
# ==============================================================================
def discover_available_examples(examples_root: Path) -> List[dict]:
    """
    Scans the `examples/` directory recursively to find all source files.
    """
    found = []
    for path in examples_root.rglob("*.cc"):
        rel_path = path.relative_to(examples_root)

        file_dir = rel_path.parent
        file_we = rel_path.stem

        category = str(file_dir).replace(os.sep, "/") if str(file_dir) != "." else ""

        binary_rel_path = f"{category}/{file_we}" if category else file_we

        found.append(
            {
                "absolute_path": path,
                "relative_path": str(rel_path).replace(os.sep, "/"),
                "category": category,
                "name_we": file_we,
                "binary_path": binary_rel_path,
            }
        )
    return found


def resolve_targets(input_strings: List[str], available: List[dict]) -> List[dict]:
    """
    Resolves user-provided strings into targets.
    Supports filtering by:
      1. Category directory (e.g. 'mpi') -> Runs everything inside
      2. Exact relative path (e.g. 'mpi/all_reduce')
      3. Short program name matching (e.g. 'all_reduce') -> Runs all instances
    """
    resolved = []
    seen_binaries = set()

    for query in input_strings:
        query = query.strip().replace(os.sep, "/")
        if not query:
            continue

        matched_any = False
        for item in available:
            if (
                query == item["category"]
                or query == item["relative_path"]
                or query == item["binary_path"]
                or query == item["name_we"]
            ):
                if item["binary_path"] not in seen_binaries:
                    seen_binaries.add(item["binary_path"])
                    resolved.append(item)
                matched_any = True

        if not matched_any:
            print(
                f"⚠️  Warning: Input pattern '{query}' did not match any discovered examples."
            )

    return resolved


# ==============================================================================
# ENGINE RUNNER
# ==============================================================================
def setup_log_directory(directory: str) -> str:
    """Creates a unique timestamped subdirectory inside the log folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_path = os.path.join(directory, f"run_{timestamp}")
    os.makedirs(target_path, exist_ok=True)
    return target_path


def run_iccl_example(
    target_info: dict,
    config_path: str,
    launcher_opt: Optional[str],
    executable_args: List[str],
    log_dir: str,
    trigger_build: bool,
    verbose: bool,
    timeout_duration: int,
) -> bool:
    """Executes an example via `icclrun` orchestration framework."""
    binary_path = target_info["binary_path"]
    safe_log_name = binary_path.replace("/", "_")
    log_file_path = os.path.join(log_dir, f"{safe_log_name}.log")

    status_msg = (
        "🚀 Running via `icclrun` (with build):"
        if trigger_build
        else "🚀 Running via `icclrun`:             "
    )
    print(f"{status_msg:<35} {binary_path:<30}", end="", flush=True)

    # Base Command Assembly
    cmd = ["icclrun", "--config", config_path]
    if launcher_opt and launcher_opt.strip():
        cmd.extend(["--launcher", launcher_opt.strip()])

    if trigger_build:
        cmd.extend(["--build", binary_path])
    else:
        cmd.append(binary_path)

    cmd.extend(executable_args)

    # Force environment unbuffered stream states for sub-python instances.
    custom_env = os.environ.copy()
    custom_env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_file_path, "w", buffering=1) as log_file:
            log_file.write(f"[COMMAND]: {' '.join(cmd)}\n")
            log_file.write("=" * 80 + "\n\n")
            log_file.flush()

            if verbose:
                print(f"\n--- [VERBOSE OUTPUT START: `{binary_path}`] ---")

                # Execute with `Popen` to stream `stdout`/`stderr` live to both terminal and file.
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=custom_env,
                    text=True,
                )

                for line in process.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)

                process.wait(timeout=timeout_duration)
                return_code = process.returncode
                print(
                    f"--- [VERBOSE OUTPUT END: `{binary_path}`] ---\n" + " " * 66,
                    end="",
                )
            else:
                # Quiet mode: Redirect straight to the file handle.
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=custom_env,
                    text=True,
                    timeout=timeout_duration,
                )
                return_code = result.returncode

        if return_code == 0:
            print(f"  ✓ PASSED  (Log saved to `{os.path.basename(log_file_path)}`)")
            return True
        else:
            print(f" ❌ FAILED  (Exit Code: {return_code})")
            return False

    except subprocess.TimeoutExpired:
        print(f" ❌ TIMEOUT (Exceeded {timeout_duration} seconds)")
        with open(log_file_path, "a") as f:
            f.write(
                f"\n[RUNNER ERROR]: Harness timed out after {timeout_duration} seconds.\n"
            )
        return False
    except FileNotFoundError:
        print(" ❌ ERROR   (`icclrun` executable not found in `PATH`)")
        return False
    except Exception as e:
        print(f" ❌ CRASHED ({str(e)})")
        return False


def main():
    # Setup CLI Argument Parsing
    parser = argparse.ArgumentParser(
        description="InfiniCCL Example Suite Execution Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="examples/cluster.yaml",
        help="Path to the cluster topology YAML configuration file (i.e., `cluster.yaml`).",
    )
    parser.add_argument(
        "-l",
        "--launcher",
        type=str,
        default=None,
        help="Specify the target launcher. If omitted, defaults to `icclrun`'s default. See `icclrun --help` for valid options.",
    )
    parser.add_argument(
        "-e",
        "--examples",
        type=str,
        default="mpi",
        help="Comma-separated paths, categories, or short names (fuzzy match). E.g. 'mpi', 'mpi/all_reduce', or 'all_reduce'.",
    )
    parser.add_argument(
        "-o",
        "--log-path",
        type=str,
        default="./example_logs",
        help="Destination directory where output logs will be stored.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=300,
        help="Timeout duration in seconds for each individual example execution pass.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose mode: stream execution `stdout`/`stderr` directly to terminal while logging.",
    )
    parser.add_argument(
        "executable_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to each selected executable. Use `--` before these args, e.g. `-- -g 4 -n 1024`.",
    )

    args = parser.parse_args()
    executable_args = args.executable_args
    if executable_args and executable_args[0] == "--":
        executable_args = executable_args[1:]

    script_dir = Path(__file__).parent.resolve()
    examples_root = script_dir / "../examples"
    if not examples_root.exists():
        examples_root = Path("examples").resolve()

    if not examples_root.exists():
        print("❌ Error: Could not locate 'examples/' directory tree.")
        sys.exit(1)

    all_available = discover_available_examples(examples_root)

    input_queries = [ex.strip() for ex in args.examples.split(",") if ex.strip()]
    targets_to_run = resolve_targets(input_queries, all_available)

    if not os.path.exists(args.config):
        print(f"❌ Error: Config file matching '{args.config}' could not be located.")
        sys.exit(1)

    if not targets_to_run:
        print(
            "❌ Error: No valid targets resolved. Please check your `--examples` query configurations."
        )
        sys.exit(1)

    # Initialize Log Infrastructure
    current_run_log_dir = setup_log_directory(args.log_path)
    launcher_display = args.launcher if args.launcher else "Internal default"

    print("==================================================================")
    print("               InfiniCCL Distributed Verification                 ")
    print(f"Target Configuration File: {args.config}")
    print(f"Selected Launcher Engine : {launcher_display}")
    print(f"Logs Target Destination  : {current_run_log_dir}")
    print(f"Verbose Console Output   : {'ENABLED' if args.verbose else 'DISABLED'}")
    print("==================================================================")

    passed_count = 0
    failed_count = 0
    failed_programs = []
    is_first_run = True

    for target in targets_to_run:
        success = run_iccl_example(
            target_info=target,
            config_path=args.config,
            launcher_opt=args.launcher,
            executable_args=executable_args,
            log_dir=current_run_log_dir,
            trigger_build=is_first_run,
            verbose=args.verbose,
            timeout_duration=args.timeout,
        )

        is_first_run = False

        if success:
            passed_count += 1
        else:
            failed_count += 1
            failed_programs.append(target["binary_path"])

    total_programs = len(targets_to_run)
    success_rate = (passed_count / total_programs) * 100

    print("\n==================================================================")
    print("                      EXECUTION SUMMARY                           ")
    print("==================================================================")
    print(f"Total Framework Targets  : {total_programs}")
    print(f"Successfully Passed      : {passed_count}")
    print(f"Failed / Crashed Targets : {failed_count}")
    print(f"Overall Success Rate     : {success_rate:.2f}%")
    print("==================================================================")

    if failed_count > 0:
        print(
            f"⚠️  The following collective validation targets failed: {', '.join(failed_programs)}"
        )
        sys.exit(1)
    else:
        print("🎉 All targets executed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
