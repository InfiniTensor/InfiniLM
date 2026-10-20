"""Prepare the pinned FlashAttention forward headers for native inference.

Apply the reviewed adaptation only in build staging, never in upstream sources.
The prepared parameter layout must not be linked to the upstream Torch extension.
"""

import argparse
import pathlib
import subprocess
import tempfile


# Explicit forward closure for the pinned source. num_splits.h is extracted
# separately from flash_api.cpp; CUTLASS headers are included directly by CMake.
HEADERS = (
    "block_info.h",
    "dropout.h",
    "flash.h",
    "flash_fwd_kernel.h",
    "flash_fwd_launch_template.h",
    "hardware_info.h",
    "kernel_traits.h",
    "mask.h",
    "namespace_config.h",
    "philox.cuh",
    "rotary.h",
    "softmax.h",
    "static_switch.h",
    "utils.h",
)
PATCH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "patches/flash-attention/inference.patch"
)


def write_if_changed(path, content):
    # Preserve timestamps across reconfiguration when prepared bytes agree.
    content = content.encode("utf-8") if isinstance(content, str) else content
    if not path.exists() or path.read_bytes() != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def read_source(path):
    if not path.is_file():
        raise ValueError(
            f"missing FlashAttention source dependency: {path}; "
            "check the fixed source archives or FetchContent SOURCE_DIR overrides"
        )
    return path.read_text(encoding="utf-8")


def prepare(source, cutlass_source, output, patch_executable="patch"):
    source, output = pathlib.Path(source).resolve(), pathlib.Path(output).resolve()
    cutlass_source = pathlib.Path(cutlass_source).resolve()
    if output.is_relative_to(source) or output.is_relative_to(cutlass_source):
        raise ValueError(
            "FlashAttention preparation output must be outside upstream source trees"
        )
    kernel_source = source / "csrc/flash_attn/src"
    headers = {name: read_source(kernel_source / name) for name in HEADERS}
    api = read_source(source / "csrc/flash_attn/flash_api.cpp")
    notices = {name: read_source(source / name) for name in ("LICENSE", "AUTHORS")}
    notices["CUTLASS-LICENSE.txt"] = read_source(cutlass_source / "LICENSE.txt")
    if not PATCH.is_file():
        raise ValueError(f"missing FlashAttention inference adaptation: {PATCH}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="flash-inference-", dir=output.parent
    ) as temp:
        staging = pathlib.Path(temp)
        for name, text in headers.items():
            write_if_changed(staging / name, text)
        try:
            applied = subprocess.run(
                [
                    str(patch_executable),
                    "--batch",
                    "--forward",
                    "--fuzz=0",
                    "-p1",
                    "--input",
                    str(PATCH),
                ],
                cwd=staging,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "native FlashAttention preparation requires the patch executable"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError("FlashAttention inference patch timed out") from exc
        if applied.returncode:
            raise ValueError(
                "pinned FlashAttention inference patch failed:\n"
                + applied.stdout
                + applied.stderr
            )
        headers = {
            name: (staging / name).read_text(encoding="utf-8") for name in HEADERS
        }

    # Keep the upstream scheduling heuristic, without introducing another policy.
    begin = api.index("inline int num_splits_heuristic(")
    end = api.index("\nstd::tuple<at::Tensor, at::Tensor> set_params_splitkv", begin)
    headers["num_splits.h"] = (
        "// Copyright (c) 2023, Tri Dao. See accompanying LICENSE.\n"
        "// Extracted from flash_attn-2.8.3/csrc/flash_attn/flash_api.cpp;\n"
        "// modified by InfiniOps to remove Torch dependencies and use a private namespace.\n"
        "#pragma once\n#include <algorithm>\n#include <cmath>\n#include <vector>\n"
        "namespace infini_flash {\n" + api[begin:end] + "\n}\n"
    )
    for name, text in {**headers, **notices}.items():
        write_if_changed(output / name, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--cutlass-source-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-executable", default="patch")
    args = parser.parse_args()
    try:
        prepare(
            args.source_dir, args.cutlass_source_dir, args.output, args.patch_executable
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"FlashAttention preparation failed: {exc}\n")
