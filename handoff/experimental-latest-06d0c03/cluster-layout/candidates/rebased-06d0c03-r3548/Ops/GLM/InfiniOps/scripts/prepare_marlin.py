#!/usr/bin/env python3
"""Prepare the pinned vLLM Marlin W4A16 sources in the build directory."""

import argparse
import pathlib
import subprocess
import sys
import tempfile


REPO = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = pathlib.Path("csrc/libtorch_stable/quantization/marlin")
SOURCES = {
    pathlib.Path(name): UPSTREAM / name
    for name in (
        "dequant.h",
        "kernel.h",
        "marlin.cuh",
        "marlin_dtypes.cuh",
        "marlin_mma.h",
        "marlin_template.h",
        "awq_marlin_repack.cu",
        "gptq_marlin_repack.cu",
        "generate_kernels.py",
        "marlin.cu",
    )
}
SOURCES[pathlib.Path("core/scalar_type.hpp")] = pathlib.Path(
    "csrc/core/scalar_type.hpp"
)
GENERATED = (
    "kernel_selector.h",
    "sm80_kernel_float16_u4_float16.cu",
    "sm80_kernel_float16_u4b8_float16.cu",
    "sm80_kernel_bfloat16_u4_bfloat16.cu",
    "sm80_kernel_bfloat16_u4b8_bfloat16.cu",
)


def write_if_changed(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)


def prepare(source, output, patch_executable):
    if output.is_relative_to(source):
        raise ValueError("Marlin output must be outside the upstream source tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="marlin-", dir=output.parent) as temporary:
        staging = pathlib.Path(temporary)
        for destination, origin in SOURCES.items():
            upstream = source / origin
            if not upstream.is_file():
                raise ValueError(
                    f"Missing pinned vLLM Marlin source: {upstream}; "
                    "check the pinned source archive or offline FetchContent override"
                )
            target = staging / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                upstream.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
            )
        applied = subprocess.run(
            [
                str(patch_executable),
                "--batch",
                "--forward",
                "--fuzz=0",
                "-p1",
                "--input",
                str(REPO / "patches/marlin/w4a16.patch"),
            ],
            cwd=staging,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if applied.returncode:
            raise ValueError(
                "Pinned Marlin adaptation failed:\n" + applied.stdout + applied.stderr
            )
        # The upstream generator's single result/config list emits both CUDA
        # instantiations and the selector. The patch retains only its W4A16 set.
        subprocess.run(
            [sys.executable, str(staging / "generate_kernels.py"), "8.0"],
            cwd=staging,
            check=True,
        )
        prepared = [path for path in SOURCES if path.suffix != ".py"]
        prepared.extend(pathlib.Path(name) for name in GENERATED)
        for destination in prepared:
            write_if_changed(output / destination, (staging / destination).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--patch-executable", default="patch")
    args = parser.parse_args()
    try:
        prepare(args.source_dir.resolve(), args.output.resolve(), args.patch_executable)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
