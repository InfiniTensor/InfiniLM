#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

BS = "16"
INPUT_LEN = "128"
OUTPUT_LEN = "128"
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

CORE_LIB = os.environ.get("INFINICORE_LIB_DIR", "/home/libaoming/workplace/InfiniCore_latest/python/infinicore/lib")
os.environ["LD_LIBRARY_PATH"] = CORE_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")

cmd = [
    sys.executable,
    "examples/bench.py",
    "--device", os.environ.get("DSV2_INFINI_DEVICE", "hygon"),
    "--use-mla",
    "--weight-load", os.environ.get("DSV2_INFINI_WEIGHT_LOAD", "sync"),
    "--model", os.environ.get("MODEL", "/home_aclsylqidf/shared/DeepSeek-V2-Lite-Chat"),
    "--batch-size", BS,
    "--input-len", INPUT_LEN,
    "--output-len", OUTPUT_LEN,
]

if USE_TP2:
    cmd += ["--tp", os.environ.get("DSV2_INFINI_TP", "2")]
else:
    # Single-GPU cases intentionally omit --tp by default.
    tp = os.environ.get("DSV2_INFINI_TP")
    if tp:
        cmd += ["--tp", tp]

if os.environ.get("DSV2_INFINI_GRAPH", "0") == "1":
    cmd.append("--enable-graph")

if os.environ.get("DSV2_WARMUP", "0") == "1":
    cmd.append("--warmup")

print("CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", ""), flush=True)
print("HIP_VISIBLE_DEVICES=" + os.environ.get("HIP_VISIBLE_DEVICES", ""), flush=True)
print(" ".join(cmd), flush=True)
raise SystemExit(subprocess.call(cmd))
