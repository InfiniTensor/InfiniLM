import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPERATOR_SET = PROJECT_ROOT / "scripts/configs/infiniops_ops.txt"
SUBMODULES = {
    "InfiniRT": Path("submodules/InfiniRT"),
    "InfiniOps": Path("submodules/InfiniOps"),
    "InfiniCCL": Path("submodules/InfiniCCL"),
}


def read_operator_set(path: Path = OPERATOR_SET) -> List[str]:
    operators = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    operators = [operator for operator in operators if operator]
    if not operators:
        raise ValueError(f"Operator set is empty: {path}")
    if len(operators) != len(set(operators)):
        raise ValueError(f"Operator set contains duplicates: {path}")
    if operators != sorted(operators):
        raise ValueError(f"Operator set must be sorted: {path}")
    return operators


def parse_gitlink(output: str, relative_path: Path) -> str:
    fields = output.strip().split()
    if len(fields) != 4 or fields[0] != "160000" or fields[1] != "commit":
        raise RuntimeError(f"Not a gitlink in HEAD: {relative_path}")
    return fields[2]


def capture(command: Sequence[str], cwd: Path) -> str:
    process = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"Command failed: {shlex.join(command)}\n{detail}")
    return process.stdout.strip()


def git_capture(arguments: Sequence[str], cwd: Path) -> str:
    return capture(
        ["git", "-c", f"safe.directory={cwd}", *arguments],
        cwd,
    )


def validate_submodule(project_root: Path, relative_path: Path) -> str:
    submodule_root = project_root / relative_path
    if not (submodule_root / "CMakeLists.txt").is_file():
        raise RuntimeError(
            f"Submodule is not initialized: {relative_path}. "
            "Run 'git submodule update --init --recursive'."
        )

    expected = parse_gitlink(
        git_capture(["ls-tree", "HEAD", str(relative_path)], project_root),
        relative_path,
    )
    actual = git_capture(["rev-parse", "HEAD"], submodule_root)
    if actual != expected:
        raise RuntimeError(
            f"Submodule revision mismatch for {relative_path}: "
            f"expected {expected}, found {actual}"
        )
    status = git_capture(["status", "--porcelain"], submodule_root)
    if status:
        raise RuntimeError(f"Submodule worktree is dirty: {relative_path}")
    return actual


def run(
    command: Sequence[str],
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def cmake_cuda_architectures(cuda_arch: str) -> str:
    return ";".join(arch[3:] for arch in cuda_arch.split(","))


def build_infinirt_commands(
    source: Path,
    build: Path,
    prefix: Path,
    build_type: str,
    jobs: int,
    cuda_arch: Optional[str],
    test: bool,
) -> List[List[str]]:
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DWITH_CPU=ON",
        "-DWITH_NVIDIA=ON",
        f"-DINFINI_RT_BUILD_TESTING={'ON' if test else 'OFF'}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
    ]
    if cuda_arch:
        configure.append(
            f"-DCMAKE_CUDA_ARCHITECTURES={cmake_cuda_architectures(cuda_arch)}"
        )

    commands = [
        configure,
        ["cmake", "--build", str(build), "--parallel", str(jobs)],
    ]
    if test:
        commands.append(
            [
                "ctest",
                "--test-dir",
                str(build),
                "--output-on-failure",
                "--parallel",
                str(jobs),
            ]
        )
    commands.append(["cmake", "--install", str(build)])
    return commands


def build_infiniops_commands(
    source: Path,
    build: Path,
    prefix: Path,
    build_type: str,
    jobs: int,
    cuda_arch: Optional[str],
    operators: Sequence[str],
) -> List[List[str]]:
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DWITH_CPU=ON",
        "-DWITH_NVIDIA=ON",
        "-DAUTO_DETECT_DEVICES=OFF",
        "-DAUTO_DETECT_BACKENDS=OFF",
        "-DGENERATE_PYTHON_BINDINGS=OFF",
        f"-DINFINI_RT_ROOT={prefix}",
        f"-DINFINI_OPS_OPS={','.join(operators)}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
    ]
    if cuda_arch:
        configure.append(
            f"-DCMAKE_CUDA_ARCHITECTURES={cmake_cuda_architectures(cuda_arch)}"
        )

    return [
        configure,
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "infiniops",
            "--parallel",
            str(jobs),
        ],
        ["cmake", "--install", str(build)],
    ]


def build_infiniccl_commands(
    source: Path,
    build: Path,
    prefix: Path,
    build_type: str,
    jobs: int,
    cuda_arch: Optional[str],
    test: bool,
) -> List[List[str]]:
    configure = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DWITH_NVIDIA=ON",
        "-DWITH_NCCL=ON",
        "-DWITH_OMPI=OFF",
        "-DWITH_MPICH=OFF",
        "-DAUTO_DETECT_DEVICES=OFF",
        "-DAUTO_DETECT_BACKENDS=OFF",
        f"-DBUILD_EXAMPLES={'ON' if test else 'OFF'}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
    ]
    if cuda_arch:
        configure.append(
            f"-DCMAKE_CUDA_ARCHITECTURES={cmake_cuda_architectures(cuda_arch)}"
        )

    commands = [
        configure,
        ["cmake", "--build", str(build), "--parallel", str(jobs)],
        ["cmake", "--install", str(build)],
    ]
    if test:
        commands.append(
            [
                str(build / "examples/ccl/all_reduce"),
                "-g",
                "2",
                "-w",
                "1",
                "-p",
                "1",
                "-n",
                "1048576",
            ]
        )
    return commands


def integration_environment(prefix: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [str(prefix / "lib"), env.get("LD_LIBRARY_PATH", "")])
    )
    return env


def manifest_staging_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def write_manifest(
    path: Path,
    infinilm_revision: str,
    infinicore_revision: str,
    revisions: Mapping[str, str],
    operators: Sequence[str],
    prefix: Path,
    build_type: str,
    cuda_arch: Optional[str],
    jobs: int,
    test: bool,
) -> None:
    data = {
        "backend": "nvidia",
        "build_type": build_type,
        "cuda_arch": cuda_arch,
        "infinicore": infinicore_revision,
        "infinilm": infinilm_revision,
        "install_prefix": str(prefix),
        "jobs": jobs,
        "operators": list(operators),
        "submodules": dict(revisions),
        "test": test,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = manifest_staging_path(path)
    try:
        staging_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        staging_path.replace(path)
    finally:
        staging_path.unlink(missing_ok=True)


def parse_cuda_arch(value: str) -> str:
    architectures = value.split(",")
    if not all(
        architecture and re.fullmatch(r"sm_\d{2}a?", architecture)
        for architecture in architectures
    ):
        raise argparse.ArgumentTypeError(
            "--cuda-arch must be a comma-separated list like sm_80,sm_86,sm_90a"
        )
    if len(architectures) != len(set(architectures)):
        raise argparse.ArgumentTypeError("--cuda-arch contains duplicates")
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the pinned InfiniRT, InfiniOps, and InfiniCCL NVIDIA stack."
    )
    parser.add_argument(
        "--infinicore-root",
        type=Path,
        required=True,
        help="InfiniCore checkout containing the pinned stack submodules.",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("build/integration/nvidia"),
        help="Isolated build and install directory (default: %(default)s).",
    )
    parser.add_argument("--build-type", choices=("Debug", "Release"), default="Release")
    parser.add_argument(
        "--cuda-arch",
        type=parse_cuda_arch,
        help=(
            "Optional comma-separated CUDA architectures in SM notation, "
            "for example sm_80,sm_86,sm_90a."
        ),
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run InfiniRT tests and a two-GPU InfiniCCL AllReduce smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    return args


def resolve_from_project(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if sys.platform != "linux" and not args.dry_run:
        raise RuntimeError("The integration build currently supports Linux only")

    infinicore_root = resolve_from_project(args.infinicore_root)
    build_root = resolve_from_project(args.build_root)
    prefix = build_root / "prefix"
    manifest_path = build_root / "manifest.json"
    operators = read_operator_set()
    revisions = {
        name: validate_submodule(infinicore_root, relative_path)
        for name, relative_path in SUBMODULES.items()
    }
    infinilm_revision = git_capture(["rev-parse", "HEAD"], PROJECT_ROOT)
    infinicore_revision = git_capture(["rev-parse", "HEAD"], infinicore_root)
    build_env = os.environ.copy()

    print(f"InfiniLM: {infinilm_revision}")
    print(f"InfiniCore: {infinicore_revision}")
    print(f"InfiniRT: {revisions['InfiniRT']}")
    print(f"InfiniOps: {revisions['InfiniOps']}")
    print(f"InfiniCCL: {revisions['InfiniCCL']}")
    print(f"Operators ({len(operators)}): {','.join(operators)}")

    if not args.dry_run:
        manifest_path.unlink(missing_ok=True)
        manifest_staging_path(manifest_path).unlink(missing_ok=True)

    for command in build_infinirt_commands(
        infinicore_root / SUBMODULES["InfiniRT"],
        build_root / "infinirt",
        prefix,
        args.build_type,
        args.jobs,
        args.cuda_arch,
        args.test,
    ):
        run(command, PROJECT_ROOT, build_env, args.dry_run)

    integration_env = integration_environment(prefix)

    for command in build_infiniops_commands(
        infinicore_root / SUBMODULES["InfiniOps"],
        build_root / "infiniops",
        prefix,
        args.build_type,
        args.jobs,
        args.cuda_arch,
        operators,
    ):
        run(command, PROJECT_ROOT, integration_env, args.dry_run)

    for command in build_infiniccl_commands(
        infinicore_root / SUBMODULES["InfiniCCL"],
        build_root / "infiniccl",
        prefix,
        args.build_type,
        args.jobs,
        args.cuda_arch,
        args.test,
    ):
        run(command, PROJECT_ROOT, integration_env, args.dry_run)

    if not args.dry_run:
        write_manifest(
            manifest_path,
            infinilm_revision,
            infinicore_revision,
            revisions,
            operators,
            prefix,
            args.build_type,
            args.cuda_arch,
            args.jobs,
            args.test,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
