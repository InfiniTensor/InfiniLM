#!/usr/bin/env python3
"""
Verify a converted MiniCPMV hybrid compressor bin against the source .pth.

This checks that:
  - header is valid
  - for each layer, the 6 weight blocks correspond to:
      compress_tk (3 weights sorted by original slot), then compress_tv (3 weights sorted by original slot)
  - reports max diff per weight

Usage:
  python scripts/verify_minicpmv_hybrid_mlp_bin.py --pth Fastcache/ckpt/minicpmv_hybrid_mlp.pth --bin minicpmv_hybrid_mlp.bin --dtype fp16
"""

import argparse
import struct
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

PREFIX_ORDER = ["compress_tk", "compress_tv"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    return ap.parse_args()


def dtype_np(dtype: str):
    return {"fp16": np.float16, "fp32": np.float32}.get(dtype)


def dtype_torch(dtype: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]

def bf16_to_f32_np(u16: np.ndarray) -> np.ndarray:
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


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


def group_weight_keys(flat: Dict[str, torch.Tensor]):
    weights_by_layer_prefix: Dict[Tuple[int, str], List[Tuple[int, torch.Tensor]]] = defaultdict(list)
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
        weights_by_layer_prefix[(layer, prefix)].append((slot, tensor))
    return weights_by_layer_prefix


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
    compressor_keys = {k: v for k, v in flat.items() if "compressor." in k}
    if compressor_keys:
        flat = {k.split("compressor.", 1)[1]: v for k, v in compressor_keys.items()}

    weights_by = group_weight_keys(flat)
    if not weights_by:
        raise RuntimeError("No expected weight keys found under compressor subtree")

    max_layer = max(layer for (layer, _) in weights_by.keys())
    num_layers = max_layer + 1

    expected_order: List[Tuple[str, int, int, torch.Tensor]] = []
    for layer in range(num_layers):
        for prefix in PREFIX_ORDER:
            items = sorted(weights_by.get((layer, prefix), []), key=lambda x: x[0])
            if len(items) != 3:
                raise RuntimeError(f"Expected 3 weights for {prefix} at layer {layer}, got {len(items)}")
            for slot, tensor in items:
                expected_order.append((prefix, layer, slot, tensor))

    with open(args.bin, "rb") as f:
        hdr = read_header(f)
        print("Header:", hdr)
        if hdr["magic"] != 0x4B56434D or hdr["version"] != 1:
            raise RuntimeError("Invalid magic/version")
        if hdr["num_layers"] != num_layers:
            raise RuntimeError(f"Layer count mismatch: bin={hdr['num_layers']} pth={num_layers}")
        if hdr["weight_count_per_layer"] != 6:
            raise RuntimeError(f"Unexpected weight_count_per_layer: {hdr['weight_count_per_layer']} (expected 6)")
        if hdr["metadata_size_bytes"] > 0:
            f.seek(hdr["metadata_size_bytes"], 1)

        np_dtype = dtype_np(args.dtype)
        torch_dtype = dtype_torch(args.dtype)
        diffs = []
        for idx, (prefix, layer, slot, t_ref) in enumerate(expected_order):
            meta_raw = f.read(12)
            if len(meta_raw) != 12:
                raise RuntimeError(f"Unexpected EOF at weight {idx}")
            rows, cols, has_bias = struct.unpack("<I I I", meta_raw)
            if args.dtype == "bf16":
                w_bytes = rows * cols * 2
                wbuf = f.read(w_bytes)
                if len(wbuf) != w_bytes:
                    raise RuntimeError(f"Unexpected EOF reading weight data at {idx}")
                w_u16 = np.frombuffer(wbuf, dtype=np.uint16).reshape(rows, cols)
                w_bin = bf16_to_f32_np(w_u16)
                w_ref = t_ref.detach().cpu().to(dtype=torch.bfloat16)
                w_ref = w_ref.view(torch.uint16).numpy()
                w_ref = bf16_to_f32_np(w_ref.reshape(rows, cols))
            else:
                if np_dtype is None:
                    raise RuntimeError(f"Unsupported dtype: {args.dtype}")
                w_bytes = rows * cols * np.dtype(np_dtype).itemsize
                wbuf = f.read(w_bytes)
                if len(wbuf) != w_bytes:
                    raise RuntimeError(f"Unexpected EOF reading weight data at {idx}")
                w_bin = np.frombuffer(wbuf, dtype=np_dtype).reshape(rows, cols).astype(np.float32)
                w_ref = t_ref.detach().cpu().to(dtype=torch_dtype).numpy().astype(np.float32)
            if w_ref.shape != (rows, cols):
                raise RuntimeError(
                    f"Shape mismatch at {prefix}.{layer}.{slot}: pth={w_ref.shape} bin={(rows, cols)}"
                )
            diff = float(np.max(np.abs(w_ref - w_bin)))
            diffs.append(diff)
            if has_bias:
                if args.dtype == "bf16":
                    _ = f.read(rows * 2)
                else:
                    _ = f.read(rows * np.dtype(np_dtype).itemsize)

        print(f"Compared {len(expected_order)} weight blocks, max diff={max(diffs):.6f}, mean diff={float(np.mean(diffs)):.6f}")
        tail = f.read()
        if tail:
            print(f"[WARN] trailing bytes: {len(tail)}")


if __name__ == "__main__":
    main()
