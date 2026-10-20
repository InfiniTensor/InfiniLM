#!/usr/bin/env python3
import argparse
import os
import sys


def configure_system_paths():
    """Dynamically resolves and injects necessary framework search paths."""
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(SCRIPT_DIR)

    # `CMAKE_INSTALL_PREFIX`
    PREFIX_DIR = os.path.dirname(SCRIPT_DIR)
    lib_found = False

    for lib_dir_name in ["lib64", "lib"]:
        candidate_path = os.path.join(PREFIX_DIR, lib_dir_name, "infiniccl")
        if os.path.exists(candidate_path):
            sys.path.append(candidate_path)
            lib_found = True
            break

    # Fallback
    if not lib_found:
        if os.path.exists(os.path.join(PREFIX_DIR, "icclrun_logic.py")):
            sys.path.append(PREFIX_DIR)
        else:
            print(
                "[Error]: Could not locate 'icclrun_logic.py' in system library paths or local workspace.",
                file=sys.stderr,
            )
            print(
                f"Looked under: {os.path.join(PREFIX_DIR, 'lib64/infiniccl')}, {os.path.join(PREFIX_DIR, 'lib/infiniccl')}, and {PREFIX_DIR}",
                file=sys.stderr,
            )
            sys.exit(1)


def main():
    configure_system_paths()

    from icclrun_logic import ICCLLauncher

    parser = argparse.ArgumentParser(description="InfiniCCL Unified Launcher")
    parser.add_argument("--config", "-c", dest="cluster", help="Path to cluster.yaml")
    parser.add_argument("--build", action="store_true", help="Compile remote nodes")
    parser.add_argument(
        "--launcher",
        choices=["ompi", "mpich", "none"],
        default="ompi",
        help="Orchestration layer: 'ompi'/'mpich' for mpirun cluster apps, 'none' for native threaded single-node apps.",
    )

    launcher_args, remaining = parser.parse_known_args()

    if not remaining:
        print("Error: No executable specified.")
        sys.exit(1)

    # The first 'remaining' item is our binary, the rest are its arguments.
    executable = remaining[0]
    app_args = remaining[1:]

    launcher = ICCLLauncher(launcher_args.cluster)

    if launcher_args.build:
        launcher.orchestrate_build()

    launcher.launch(launcher_args.launcher, executable, app_args, launcher)


if __name__ == "__main__":
    main()
