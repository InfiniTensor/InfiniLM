"""Prepare pinned upstream forward kernels without a Torch runtime dependency.

This is a build-time, inference-only adaptation, not another Attention algorithm.
Do not link these altered parameter layouts to the original Torch extension.
"""

import argparse
import hashlib
import json
import pathlib
import re
import tarfile


ARCHIVE_SHA256 = "1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d"
ROOT = "flash_attn-2.8.3/"
KERNEL_ROOT = ROOT + "csrc/flash_attn/src/"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"pinned FlashAttention source contract changed: {before!r}")
    return text.replace(before, after, 1)


def write_if_changed(path, content):
    # CMake may prepare the same pinned closure again during an incremental
    # build. Preserve timestamps so hundreds of CUDA kernels are not rebuilt
    # merely because configure ran; changed content still invalidates them.
    content = content.encode("utf-8") if isinstance(content, str) else content
    if not path.exists() or path.read_bytes() != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def adapt(name, text):
    if name == "flash.h":
        text = replace_once(text, "#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState", "")
        text = replace_once(text, "    at::PhiloxCudaState philox_args;", "    // Inference-only build: dropout and its Torch RNG state are absent.")
    elif name == "flash_fwd_kernel.h":
        text = replace_once(text, '#include "philox_unpack.cuh" // For at::cuda::philox::unpack', "")
        text = replace_once(text, "    auto seed_offset = at::cuda::philox::unpack(params.philox_args);", "    static_assert(!Is_dropout, \"InfiniOps inference build does not support dropout\");\n    const auto seed_offset = std::make_tuple(uint64_t{0}, uint64_t{0});")
    elif name == "flash_fwd_launch_template.h":
        text = replace_once(text, "#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK", '#include "infini_flash_cuda.h"')
        text = text.replace("C10_CUDA_KERNEL_LAUNCH_CHECK", "INFINI_FLASH_KERNEL_CHECK")
        text = text.replace("C10_CUDA_CHECK", "INFINI_FLASH_CUDA_CHECK")
    elif name == "hardware_info.h":
        text = replace_once(text, "#include <tuple>", '#include <tuple>\n#include "infini_flash_cuda.h"')
        text = replace_once(text, "      exit(1);", "      INFINI_FLASH_CUDA_CHECK(status_);")
    return text


def prepare(archive, output):
    archive, output = pathlib.Path(archive), pathlib.Path(output)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"expected pinned flash_attn-2.8.3 archive {ARCHIVE_SHA256}, got {digest}")
    with tarfile.open(archive, "r:gz") as source:
        members = {member.name: member for member in source.getmembers()}

        def read(name):
            member = members.get(name)
            if member is None or not member.isfile():
                raise ValueError(f"missing regular pinned source file: {name}")
            return source.extractfile(member).read()

        # Only the transitive forward include closure is needed. No backward,
        # ATen wrapper, Python extension or Torch allocator is compiled.
        pending = ["flash_fwd_launch_template.h"]
        headers = {}
        while pending:
            name = pending.pop()
            if name in headers:
                continue
            if pathlib.PurePosixPath(name).name != name:
                raise ValueError(f"unexpected upstream include: {name}")
            text = adapt(name, read(KERNEL_ROOT + name).decode())
            if re.search(r"#\s*include\s*[<\"](?:ATen|c10|torch)/|\bat::|\bc10::", text):
                raise ValueError(f"unremoved Torch dependency in {name}")
            headers[name] = text
            for include in re.findall(r'#\s*include\s*"([^"]+)"', text):
                if KERNEL_ROOT + include in members:
                    pending.append(include)

        # Reuse the pinned host scheduling heuristic as well as the kernels;
        # do not invent a second split-KV tuning policy in LM.
        api = read(ROOT + "csrc/flash_attn/flash_api.cpp").decode()
        begin = api.index("inline int num_splits_heuristic(")
        end = api.index("\nstd::tuple<at::Tensor, at::Tensor> set_params_splitkv", begin)
        headers["num_splits.h"] = (
            "// Copyright (c) 2023, Tri Dao. See accompanying LICENSE.\n"
            "#pragma once\n#include <algorithm>\n#include <cmath>\n#include <vector>\n"
            "namespace infini_flash {\n" + api[begin:end] + "\n}\n"
        )

        output.mkdir(parents=True, exist_ok=True)
        for name, text in headers.items():
            write_if_changed(output / name, text)
        for member in source.getmembers():
            prefix = ROOT + "csrc/cutlass/include/"
            if member.name.startswith(prefix) and member.isfile():
                relative = pathlib.PurePosixPath(member.name[len(prefix):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe CUTLASS archive path")
                path = output / "cutlass" / pathlib.Path(*relative.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                write_if_changed(path, read(member.name))
        for name in ("LICENSE", "AUTHORS"):
            write_if_changed(output / name, read(ROOT + name))
        manifest = {
            "upstream": "flash-attn", "version": "2.8.3",
            "archive_sha256": digest, "namespace": "infini_flash",
            "inference_only": True, "dropout": False,
            "headers_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                               for name, text in sorted(headers.items())},
        }
        write_if_changed(output / "source-manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output)
