#!/usr/bin/env python3
"""
Convert Fastcache MiniCPMV KVCacheHybridCompressor weights (.pth) to the binary format
defined in docs/KVCacheCompressionWeightFormat.md.

This exporter targets the minimal Hybrid path used by MiniCPMV in practice:
  - only text branches: compress_tk and compress_tv
  - 3 Linear layers per branch (ReLU/Dropout ignored)

Expected .pth key pattern under compressor subtree:
  compress_tk.<layer>.<slot>.weight / .bias (bias optional)
  compress_tv.<layer>.<slot>.weight / .bias (bias optional)

Notes:
  - The PyTorch Sequential indices are often 0/3/6; we preserve sorted-by-slot order,
    and C++ treats them as stage 0/1/2.
"""

import argparse
import struct
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

PREFIX_ORDER = ["compress_tk", "compress_tv"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to .pth file (state dict)")
    ap.add_argument("--out", required=True, help="Output binary path")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--compression-factor", type=int, default=5)
    ap.add_argument("--min-seq-len", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=0)
    ap.add_argument("--head-dim", type=int, default=0)
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
            collect_tensors(v, prefix + str(k) + ".", out)
    return out


def group_keys(flat: Dict[str, torch.Tensor]):
    """
    Group keys by (prefix, layer, slot, is_bias).
    Pattern:
      <prefix>.<layer>.<slot>.weight
      <prefix>.<layer>.<slot>.bias
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
        is_bias = kind == "bias"
        grouped.append((prefix, layer, slot, is_bias, tensor))
    return grouped


def write_header(
    f,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
    dtype: str,
    compression_factor: int,
    min_seq_len: int,
    weight_count: int,
):
    magic = 0x4B56434D  # "KV C M"
    version = 1
    reserved = 0
    metadata_size_bytes = 0
    f.write(
        struct.pack(
            "<I I H H I I I I I I I I",
            magic,
            version,
            dtype_code(dtype),
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
    )


def write_tensor_bytes(f, t: torch.Tensor):
    t = t.cpu().contiguous()
    # NumPy may not support bfloat16 directly; write raw bf16 bits instead.
    if t.dtype == torch.bfloat16:
        f.write(t.view(torch.uint16).numpy().tobytes())
        return
    f.write(t.numpy().tobytes())


def main():
    args = parse_args()
    state = torch.load(args.pth, map_location="cpu")
    flat = collect_tensors(state)
    compressor_keys = {k: v for k, v in flat.items() if "compressor." in k}
    if compressor_keys:
        flat = {k.split("compressor.", 1)[1]: v for k, v in compressor_keys.items()}

    grouped = group_keys(flat)
    if not grouped:
        raise RuntimeError(
            "No compatible weight/bias keys found (expected prefixes "
            f"{PREFIX_ORDER} and pattern <prefix>.<layer>.<slot>.weight)"
        )

    num_layers = max(g[1] for g in grouped) + 1

    # Validate that each layer has exactly 3 weights for each prefix.
    weights_by_layer_prefix: Dict[Tuple[int, str], List[Tuple[int, torch.Tensor]]] = defaultdict(list)
    bias_by_layer_prefix_slot: Dict[Tuple[int, str, int], torch.Tensor] = {}
    for prefix, layer, slot, is_bias, tensor in grouped:
        if is_bias:
            bias_by_layer_prefix_slot[(layer, prefix, slot)] = tensor
        else:
            weights_by_layer_prefix[(layer, prefix)].append((slot, tensor))

    for layer in range(num_layers):
        for prefix in PREFIX_ORDER:
            slots = sorted(s for (s, _) in weights_by_layer_prefix.get((layer, prefix), []))
            if len(slots) != 3:
                raise RuntimeError(
                    f"Expected 3 weight tensors for {prefix} at layer {layer}, got {len(slots)}: {slots}"
                )

    # Infer hidden_size from the first weight (cols).
    sample = next(t for (_, _, _, is_bias, t) in grouped if not is_bias)
    if sample.ndim != 2:
        raise RuntimeError(f"Sample tensor is not 2D: shape={tuple(sample.shape)}")
    hidden_size = int(sample.shape[1])

    weight_count_per_layer = 6  # compress_tk[3] + compress_tv[3]

    with open(args.out, "wb") as f:
        write_header(
            f,
            num_layers=num_layers,
            num_heads=int(args.num_heads),
            head_dim=int(args.head_dim),
            hidden_size=hidden_size,
            dtype=args.dtype,
            compression_factor=int(args.compression_factor),
            min_seq_len=int(args.min_seq_len),
            weight_count=weight_count_per_layer,
        )

        for layer in range(num_layers):
            for prefix in PREFIX_ORDER:
                stage_weights = sorted(weights_by_layer_prefix[(layer, prefix)], key=lambda x: x[0])
                for slot, w in stage_weights:
                    w_cast = tensor_to_dtype(w, args.dtype)
                    if w_cast.ndim != 2:
                        raise RuntimeError(f"{prefix}.{layer}.{slot}.weight is not 2D: {tuple(w_cast.shape)}")
                    rows, cols = int(w_cast.shape[0]), int(w_cast.shape[1])
                    bias = bias_by_layer_prefix_slot.get((layer, prefix, slot), None)
                    has_bias = 1 if bias is not None else 0
                    f.write(struct.pack("<I I I", rows, cols, has_bias))
                    write_tensor_bytes(f, w_cast)
                    if bias is not None:
                        b_cast = tensor_to_dtype(bias, args.dtype).reshape(-1)
                        if int(b_cast.numel()) != rows:
                            raise RuntimeError(
                                f"{prefix}.{layer}.{slot}.bias length mismatch: {int(b_cast.numel())} != {rows}"
                            )
                        write_tensor_bytes(f, b_cast)

    print(f"Wrote {args.out} (layers={num_layers}, weights/layer={weight_count_per_layer})")


if __name__ == "__main__":
    main()
