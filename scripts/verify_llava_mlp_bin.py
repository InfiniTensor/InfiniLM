#!/usr/bin/env python3
"""
Verify a converted llava_mlp.bin against the source .pth.
Checks:
  - Header fields are sane.
  - All tensors from .pth with prefixes (compress_tk, compress_tv, compress_iv, attention)
    are present in the bin (weights; bias are ignored/skipped if mismatch).
  - Reports shape mismatches and max diff for matching tensors.

Usage:
  python scripts/verify_llava_mlp_bin.py --pth Fastcache/ckpt/llava_mlp.pth --bin Fastcache/ckpt/llava_mlp.bin
"""

import argparse
import struct
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np
import torch

PREFIX_ORDER = ["compress_tk", "compress_tv", "compress_ik", "compress_iv", "attention"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Expected dtype in bin")
    return ap.parse_args()


def collect_tensors(obj, prefix: str = "", out: dict = None):
    if out is None:
        out = {}
    if isinstance(obj, torch.Tensor):
        key = prefix[:-1] if prefix.endswith(".") else prefix
        out[key] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect_tensors(v, prefix + str(k) + ".", out)
    return out


def group_keys(flat: dict):
    grouped = []
    for key, tensor in flat.items():
        parts = key.split(".")
        if len(parts) < 4:
            continue
        prefix, layer_str, slot_str, kind = parts[0], parts[1], parts[2], parts[3]
        if prefix not in PREFIX_ORDER or kind != "weight":
            continue
        try:
            layer = int(layer_str)
            slot = int(slot_str)
        except ValueError:
            continue
        grouped.append((prefix, layer, slot, tensor))
    return grouped


def dtype_np(dtype: str):
    return {"fp16": np.float16, "bf16": np.float16, "fp32": np.float32}[dtype]


def dtype_torch(dtype: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]


def read_header(f):
    hdr_fmt = "<I I H H I I I I I I I I"
    data = f.read(struct.calcsize(hdr_fmt))
    if len(data) != struct.calcsize(hdr_fmt):
        raise RuntimeError("Failed to read header")
    fields = struct.unpack(hdr_fmt, data)
    return {
        "magic": fields[0],
        "version": fields[1],
        "dtype_code": fields[2],
        "reserved": fields[3],
        "num_layers": fields[4],
        "num_heads": fields[5],
        "head_dim": fields[6],
        "hidden_size": fields[7],
        "compression_factor": fields[8],
        "min_seq_len": fields[9],
        "weight_count_per_layer": fields[10],
        "metadata_size_bytes": fields[11],
    }


def main():
    args = parse_args()
    state = torch.load(args.pth, map_location="cpu")
    flat = collect_tensors(state)
    # Focus on compressor subtree if present
    compressor_keys = {k: v for k, v in flat.items() if "compressor." in k}
    if compressor_keys:
        flat = {k.split("compressor.", 1)[1]: v for k, v in compressor_keys.items()}
    grouped = group_keys(flat)
    if not grouped:
        raise RuntimeError("No weight keys found with expected prefixes")

    expected_order = sorted(
        grouped,
        key=lambda x: (
            PREFIX_ORDER.index(x[0]),
            x[1],
            x[2],
        ),
    )

    with open(args.bin, "rb") as f:
        hdr = read_header(f)
        print("Header:", hdr)
        if hdr["magic"] != 0x4B56434D or hdr["version"] != 1:
            raise RuntimeError("Invalid magic/version")
        # Skip metadata if any
        if hdr["metadata_size_bytes"] > 0:
            f.seek(hdr["metadata_size_bytes"], 1)

        dtype = args.dtype
        np_dtype = dtype_np(dtype)
        torch_dtype = dtype_torch(dtype)
        mismatches = []
        idx = 0
        for prefix, layer, slot, tensor in expected_order:
            meta_raw = f.read(12)
            if len(meta_raw) != 12:
                raise RuntimeError(f"Unexpected EOF at weight {idx}")
            rows, cols, has_bias = struct.unpack("<I I I", meta_raw)
            weight_bytes = rows * cols * np.dtype(np_dtype).itemsize
            wbuf = f.read(weight_bytes)
            if len(wbuf) != weight_bytes:
                raise RuntimeError(f"Unexpected EOF reading weight data at {idx}")
            w_bin = np.frombuffer(wbuf, dtype=np_dtype).reshape(rows, cols)
            w_ref = tensor.detach().cpu().to(dtype=torch_dtype).numpy()
            if w_ref.shape != (rows, cols):
                mismatches.append((prefix, layer, slot, "shape", w_ref.shape, (rows, cols)))
            else:
                diff = np.max(np.abs(w_ref - w_bin))
                mismatches.append((prefix, layer, slot, "max_diff", float(diff)))

            if has_bias:
                # Skip bias bytes
                bias_bytes = rows * np.dtype(np_dtype).itemsize
                _ = f.read(bias_bytes)
            idx += 1

        # Report
        print(f"Compared {idx} weight blocks")
        shape_issues = [m for m in mismatches if m[3] == "shape"]
        diff_stats = [m for m in mismatches if m[3] == "max_diff"]
        if shape_issues:
            print("Shape mismatches:")
            for m in shape_issues[:20]:
                print(m)
        if diff_stats:
            diffs = [m[4] for m in diff_stats if isinstance(m[4], float)]
            if diffs:
                print(f"Max diff across weights: {max(diffs):.6f}")
                print(f"Mean diff across weights: {np.mean(diffs):.6f}")
        # Check EOF
        tail = f.read()
        if tail:
            print(f"[WARN] trailing bytes: {len(tail)}")


if __name__ == "__main__":
    main()
