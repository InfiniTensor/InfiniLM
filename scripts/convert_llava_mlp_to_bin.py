#!/usr/bin/env python3
"""
Convert Fastcache/ckpt/llava_mlp.pth (KVCacheLinearDecoupleCompressor weights)
to the binary format defined in docs/KVCacheCompressionWeightFormat.md.

Usage:
  python scripts/convert_llava_mlp_to_bin.py \
      --pth Fastcache/ckpt/llava_mlp.pth \
      --out ckpt/llava_mlp.bin \
      --dtype fp16 \
      --compression-factor 5 \
      --min-seq-len 2

Notes:
  - Requires PyTorch in your environment (offline/one-time step).
  - Assumes state dict keys follow the pattern used in KVCacheLinearDecoupleCompressor.
    Adjust `WEIGHT_ORDER` and key names below if they differ.
"""

import argparse
import struct
import sys
from typing import List, Tuple

import torch

# Prefix priorities to write weights in a deterministic order.
PREFIX_ORDER = ["compress_tk", "compress_tv", "compress_ik" ,"compress_iv", "attention"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to .pth file (state dict)")
    ap.add_argument("--out", required=True, help="Output binary path")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--compression-factor", type=int, default=5)
    ap.add_argument("--min-seq-len", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=None, help="Override num_heads if needed")
    ap.add_argument("--head-dim", type=int, default=None, help="Override head_dim if needed")
    return ap.parse_args()


def dtype_code(dtype: str) -> int:
    return {"fp16": 0, "bf16": 1, "fp32": 2}[dtype]


def tensor_to_dtype(t: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "fp16":
        return t.half()
    if dtype == "bf16":
        return t.bfloat16()
    if dtype == "fp32":
        return t.float()
    raise ValueError(f"Unsupported dtype: {dtype}")


def collect_tensors(obj, prefix: str = "", out: dict = None):
    if out is None:
        out = {}
    if isinstance(obj, torch.Tensor):
        key = prefix[:-1] if prefix.endswith(".") else prefix
        out[key] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key_part = str(k)
            collect_tensors(v, prefix + key_part + ".", out)
    return out


def group_keys(flat: dict):
    """
    Group keys by (prefix, layer, slot, is_bias). Expected patterns:
      - <prefix>.<layer>.<slot>.weight
      - <prefix>.<layer>.<slot>.bias  (bias optional)
    """
    grouped = []
    for key, tensor in flat.items():
        parts = key.split(".")
        if len(parts) < 4:
            continue
        prefix, layer_str, slot_str, kind = parts[0], parts[1], parts[2], parts[3]
        if prefix not in PREFIX_ORDER:
            continue
        if kind not in ("weight", "bias"):
            continue
        try:
            layer = int(layer_str)
            slot = int(slot_str)
        except ValueError:
            continue
        is_bias = (kind == "bias")
        grouped.append((prefix, layer, slot, is_bias, tensor))
    return grouped


def write_header(f, num_layers: int, num_heads: int, head_dim: int, hidden_size: int,
                 dtype: str, compression_factor: int, min_seq_len: int, weight_count: int):
    magic = 0x4B56434D  # "KV C M"
    version = 1
    dtype_code_val = dtype_code(dtype)
    reserved = 0
    metadata_size_bytes = 0
    header = struct.pack(
        "<I I H H I I I I I I I I",
        magic,
        version,
        dtype_code_val,
        reserved,
        num_layers,
        num_heads,
        head_dim,
        hidden_size,
        compression_factor,
        min_seq_len,
        weight_count,
        metadata_size_bytes,
    )
    f.write(header)


def write_tensor(f, t: torch.Tensor):
    data = t.cpu().contiguous().numpy().tobytes()
    f.write(data)


def main():
    args = parse_args()
    state = torch.load(args.pth, map_location="cpu")
    # Flatten to a dict of key -> tensor (with hierarchical prefixes).
    model_state_dict = state.get("model_state_dict", state)
    compressor = model_state_dict.get("compressor", None)
    keys = compressor.keys()
    flat = collect_tensors(state)
    if not flat:
        raise RuntimeError("No tensor found in state dict (after flatten)")

    # Try to focus on compressor subtree if present.
    # Heuristic: keys containing 'compressor.' get trimmed.
    compressor_keys = {k: v for k, v in flat.items() if "compressor." in k}
    if compressor_keys:
        flat = {k.split("compressor.", 1)[1]: v for k, v in compressor_keys.items()}

    grouped = group_keys(flat)
    if not grouped:
        raise RuntimeError("No compatible weight/bias keys found (expected prefix in "
                           f"{PREFIX_ORDER} and pattern <prefix>.<layer>.<slot>.weight)")

    # Determine layer count.
    num_layers = max(g[1] for g in grouped) + 1

    # Infer hidden_size from first weight tensor
    sample = next(t for (_, _, _, is_bias, t) in grouped if not is_bias)
    if sample.ndim < 2:
        raise RuntimeError(f"Sample tensor is not 2D: shape={sample.shape}")
    hidden_size = sample.shape[-1]

    # Optionally override num_heads/head_dim
    num_heads = args.num_heads if args.num_heads is not None else 0
    head_dim = args.head_dim if args.head_dim is not None else 0

    with open(args.out, "wb") as f:
        # Sort keys: by prefix priority, then layer, then slot, bias after weight.
        grouped_sorted = sorted(
            grouped,
            key=lambda x: (
                PREFIX_ORDER.index(x[0]) if x[0] in PREFIX_ORDER else len(PREFIX_ORDER),
                x[1],  # layer
                x[2],  # slot
                1 if x[3] else 0  # weight before bias
            ),
        )

        # Count weight entries per layer (weights only).
        weight_count = max(
            1,
            max((sum(1 for g in grouped if g[1] == layer and not g[3]) for layer in range(num_layers)))
        )

        write_header(
            f,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            dtype=args.dtype,
            compression_factor=args.compression_factor,
            min_seq_len=args.min_seq_len,
            weight_count=weight_count,
        )

        grouped_sorted_layer_wise = [None for i in range(768)]
        for i in range(32):
            grouped_sorted_layer_wise[i * 24: i*24 + 6] = grouped_sorted[i * 6: i * 6 + 6]
            grouped_sorted_layer_wise[i * 24 + 6: i*24 + 12] = grouped_sorted[32 * 6 + i * 6: 32 * 6 + i * 6 + 6]
            grouped_sorted_layer_wise[i * 24 + 12: i*24 + 18] = grouped_sorted[64 * 6 + i * 6: 64 * 6 + i * 6 + 6]
            grouped_sorted_layer_wise[i * 24 + 18: i*24 + 24] = grouped_sorted[96 * 6 + i * 6: 96 * 6 + i * 6 + 6]
        
        for prefix, layer, slot, is_bias, tensor in grouped_sorted_layer_wise:
            print(f"{prefix}.{layer}.{slot}{'.bias' if is_bias else '.weight'}: {tensor.shape}")
        
            
        

        #for prefix, layer, slot, is_bias, tensor in grouped_sorted:
        for prefix, layer, slot, is_bias, tensor in grouped_sorted_layer_wise:
            #import pdb;pdb.set_trace()
            if is_bias:
                # Bias will be written immediately after its weight.
                continue
            weight = tensor_to_dtype(tensor, args.dtype)
            rows, cols = weight.shape

            # Find bias counterpart if exists.
            bias_key = (prefix, layer, slot, True)
            bias_tensor = None
            for entry in grouped:
                if entry[0] == bias_key[0] and entry[1] == bias_key[1] and entry[2] == bias_key[2] and entry[3]:
                    bias_tensor = entry[4]
                    break
            has_bias = 1 if bias_tensor is not None else 0

            meta = struct.pack("<I I I", rows, cols, has_bias)
            f.write(meta)
            write_tensor(f, weight)
            if has_bias:
                bias = tensor_to_dtype(bias_tensor, args.dtype)
                expected = rows  # for linear layers, bias matches output dimension (rows)
                if bias.numel() != expected:
                    print(f"[WARN] Bias size mismatch for {prefix}.{layer}.{slot}.bias: expected {expected}, got {bias.numel()}; skipping bias")
                    continue
                write_tensor(f, bias)

    print(f"Wrote binary weights to {args.out} (layers={num_layers}, dtype={args.dtype})")


if __name__ == "__main__":
    main()
