#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

BS = "4"
INPUT_LEN = "1024"
OUTPUT_LEN = "1024"
USE_TP2 = False


def env_first(*names, default=None):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


visible_devices = env_first(
    "DSV2_TP2_GPUS", "TP2_GPUS", default="0,1"
) if USE_TP2 else env_first(
    "DSV2_SINGLE_GPU", "SINGLE_GPU", default="0"
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", visible_devices)
os.environ.setdefault("HIP_VISIBLE_DEVICES", visible_devices)

# CUDA/HIP_VISIBLE_DEVICES remaps selected physical devices to logical ids.
logical_devices = "0,1" if USE_TP2 else "0"
devices = os.environ.get("DSV2_TORCH_DEVICES", logical_devices)

cmd = [
    sys.executable,
    "examples/pytorch_bench.py",
    "--case-custom", BS, INPUT_LEN, OUTPUT_LEN,
    "--model", os.environ.get("MODEL", "/home_aclsylqidf/shared/DeepSeek-V2-Lite-Chat"),
    "--devices", devices,
    "--attn-implementation", os.environ.get("DSV2_TORCH_ATTN", "flash_attention_2"),
    "--last-token-logits",
]

if USE_TP2:
    cmd += [
        "--device-map", os.environ.get("DSV2_TORCH_DEVICE_MAP", "dsv2_lmhead0"),
        "--split-layer", os.environ.get("DSV2_TORCH_SPLIT_LAYER", "13"),
        "--max-memory", os.environ.get("DSV2_TORCH_MAX_MEMORY", "0:60GiB,1:60GiB"),
    ]
else:
    cmd += ["--device", os.environ.get("DSV2_TORCH_DEVICE", "cuda")]

progress = os.environ.get("DSV2_PROGRESS_STEP")
if progress:
    cmd += ["--progress-step", progress]

if os.environ.get("DSV2_WARMUP", "0") != "1":
    cmd.append("--no-warmup")

print("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)
print("HIP_VISIBLE_DEVICES=" + os.environ.get("HIP_VISIBLE_DEVICES", ""), flush=True)
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd))
