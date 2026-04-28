"""
Build orchestration for InfiniLM.

Environment knobs (read at install/build time):
  INFINILM_ENABLE_HYGON=1
      Build for Hygon DCU (DTK/HIP) instead of NVIDIA. Source DTK env first
      (e.g. `source /opt/dtk/env.sh`) so hipcc, hipblas, and rccl are findable.
      Defaults to OFF (NVIDIA build).

  INFINILM_BUILD_FLASH_ATTN=1
      Enable the FlashAttention backend. When set, flash-attention source is
      auto-cloned to third_party/flash-attention if missing, and the resulting
      libflash-attn-{nvidia,hygon}.so is co-located with libinfinicore.so.
      Under INFINILM_ENABLE_HYGON, the auto-clone default URL does NOT match a
      DTK-buildable fork; provide INFINILM_FLASH_ATTN_DIR (or a Hygon-specific
      INFINILM_FLASH_ATTN_REPO) explicitly.

  INFINILM_FLASH_ATTN_REPO=<git url>
      Git URL for the flash-attention fork.
      Default: https://github.com/vllm-project/flash-attention.git (NVIDIA-only).

  INFINILM_FLASH_ATTN_REF=<branch|tag|commit>
      Git ref to check out. Default: main.

  INFINILM_FLASH_ATTN_DIR=/abs/path
      Pre-existing flash-attention checkout to use instead of cloning. Required
      under INFINILM_ENABLE_HYGON if INFINILM_BUILD_FLASH_ATTN=1.

  INFINILM_FLASH_ATTN_ARCHS=80;86;89;90  (NVIDIA)  |  gfx906;gfx926;...  (Hygon)
      Architecture list. Default for NVIDIA: 80. Default for Hygon: full DCU set.
      Single arch shrinks the .so by ~4x.

  INFINILM_BUILD_TYPE=Release|Debug|RelWithDebInfo
      CMake build type. Default: Release.

  INFINILM_BUILD_JOBS=<int>
      Parallel compile jobs. Default: nproc.
"""

import multiprocessing
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build
from setuptools.command.develop import develop

ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
DEFAULT_FLASH_ATTN_DIR = ROOT / "third_party" / "flash-attention"


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _ensure_flash_attn_dir() -> Path | None:
    if not _env_bool("INFINILM_BUILD_FLASH_ATTN"):
        return None

    user_dir = os.environ.get("INFINILM_FLASH_ATTN_DIR")
    if user_dir:
        d = Path(user_dir).resolve()
        if not (d / "csrc" / "flash_attn" / "flash_api.cpp").is_file():
            raise RuntimeError(
                f"INFINILM_FLASH_ATTN_DIR={d} is not a flash-attention checkout"
            )
        return d

    d = DEFAULT_FLASH_ATTN_DIR
    flash_api = d / "csrc" / "flash_attn" / "flash_api.cpp"
    cutlass_h = d / "csrc" / "cutlass" / "include" / "cutlass" / "cutlass.h"

    if not flash_api.is_file():
        repo = os.environ.get(
            "INFINILM_FLASH_ATTN_REPO",
            "https://github.com/vllm-project/flash-attention.git",
        )
        ref = os.environ.get("INFINILM_FLASH_ATTN_REF", "main")
        print(f"[InfiniLM] cloning {repo}@{ref} -> {d}")
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            shutil.rmtree(d)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, repo, str(d)],
            check=True,
        )

    if not cutlass_h.is_file():
        print(f"[InfiniLM] initializing flash-attention submodules under {d}")
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=d,
            check=True,
        )
    return d


def _source_dtk_env():
    """When building for Hygon, source DTK's env scripts into os.environ so
    cmake sees ROCM_PATH, CUDA_HOME, PATH, LD_LIBRARY_PATH, *_INCLUDE_PATH.
    Idempotent: if the caller already sourced them, ROCM_PATH is set and we
    return without re-running.
    """
    if os.environ.get("ROCM_PATH"):
        return

    dtk_root = Path(os.environ.get("INFINI_HYGON_DTK_ROOT", "/opt/dtk"))
    env_sh = dtk_root / "env.sh"
    cuda_env_sh = dtk_root / "cuda" / "cuda-12" / "env.sh"

    missing = [str(p) for p in (env_sh, cuda_env_sh) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"INFINILM_ENABLE_HYGON=1 but DTK env scripts not found: {missing}. "
            f"Set INFINI_HYGON_DTK_ROOT (default /opt/dtk) or source manually."
        )

    print(f"[InfiniLM] sourcing DTK env from {dtk_root}")
    result = subprocess.run(
        [
            "bash",
            "-c",
            # source noise -> stderr; env -0 -> stdout. set -e fails fast on
            # broken env scripts; Python captures the two streams separately
            # so source output never contaminates env parsing.
            f"set -e; {{ source {env_sh}; source {cuda_env_sh}; }} 1>&2; env -0",
        ],
        capture_output=True,
        check=True,
    )
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        key, _, val = entry.decode().partition("=")
        if key:
            os.environ[key] = val


def build_cpp_module():
    BUILD_DIR.mkdir(exist_ok=True)

    enable_hygon = _env_bool("INFINILM_ENABLE_HYGON")
    if enable_hygon:
        _source_dtk_env()
    elif os.path.isdir("/opt/dtk"):
        print(
            "[InfiniLM] /opt/dtk detected but INFINILM_ENABLE_HYGON not set — "
            "building NVIDIA target. Set INFINILM_ENABLE_HYGON=1 for Hygon DCU."
        )

    cmake_args = [
        "cmake",
        "-S", str(ROOT),
        "-B", str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={os.environ.get('INFINILM_BUILD_TYPE', 'Release')}",
        f"-DINFINILM_ENABLE_HYGON={'ON' if enable_hygon else 'OFF'}",
    ]

    fa_dir = _ensure_flash_attn_dir()
    if enable_hygon and _env_bool("INFINILM_BUILD_FLASH_ATTN") and not fa_dir:
        raise RuntimeError(
            "INFINILM_ENABLE_HYGON=1 + INFINILM_BUILD_FLASH_ATTN=1 requires "
            "INFINILM_FLASH_ATTN_DIR pointing at a DTK-buildable flash-attention "
            "source fork (the same source that produced your installed "
            "flash_attn==*.dtk* wheel). The default vllm-project fork is NVIDIA-only."
        )
    # Always pass the option so we override any stale CMake cache from a
    # previous configure. Empty value disables flash-attn cleanly.
    cmake_args.append(f"-DINFINIOPS_FLASH_ATTN_DIR={fa_dir or ''}")
    cmake_args.append(
        f"-DINFINIOPS_FLASH_ATTN_ARCHS={os.environ.get('INFINILM_FLASH_ATTN_ARCHS', '')}"
    )

    subprocess.run(cmake_args, check=True)

    jobs = os.environ.get("INFINILM_BUILD_JOBS") or str(multiprocessing.cpu_count())
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "-j", jobs],
        check=True,
    )


class Build(build):
    def run(self):
        build_cpp_module()
        super().run()


class Develop(develop):
    def run(self):
        build_cpp_module()
        super().run()


setup(
    name="InfiniLM",
    version="0.1.0",
    description="InfiniLM model implementations",
    package_dir={"": "python"},
    packages=[
        "infinilm",
        "infinilm.models",
        "infinilm.lib",
        "infinilm.distributed",
        "infinicore",
        "infinicore.nn",
        "infinicore.ops",
    ],
    package_data={
        "infinilm": ["*.so", "lib/*.so"],
        "infinicore": ["*.so", "lib/*.so"],
    },
    include_package_data=True,
    cmdclass={
        "build": Build,
        "develop": Develop,
    },
    python_requires=">=3.10",
)
