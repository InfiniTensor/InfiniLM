#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


MAGIC = 0x4B56434D  # "KV C M" little endian
VERSION = 1


def _extract_compressor_state(obj: object) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")
    if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        msd = obj["model_state_dict"]
        if "compressor" in msd and isinstance(msd["compressor"], dict):
            return msd["compressor"]
    if "compressor" in obj and isinstance(obj["compressor"], dict):
        return obj["compressor"]
    if all(isinstance(k, str) for k in obj.keys()):
        # Assume it is a flat state_dict already.
        return obj  # type: ignore[return-value]
    raise ValueError("Cannot find compressor state dict in checkpoint")


def _dtype_code(dtype: str) -> Tuple[int, np.dtype]:
    dtype = dtype.lower()
    if dtype in ("fp16", "f16", "float16"):
        return 0, np.dtype("<f2")
    if dtype in ("bf16", "bfloat16"):
        # Stored as uint16 payload for bf16 (same as torch.bfloat16 storage).
        return 1, np.dtype("<u2")
    if dtype in ("fp32", "f32", "float32"):
        return 2, np.dtype("<f4")
    raise ValueError(f"Unsupported dtype: {dtype}")


def _to_numpy_bytes(x: torch.Tensor, dtype: str) -> bytes:
    if dtype in ("fp16", "f16", "float16"):
        arr = x.detach().to(dtype=torch.float16, device="cpu").contiguous().numpy().astype("<f2", copy=False)
        return arr.tobytes()
    if dtype in ("fp32", "f32", "float32"):
        arr = x.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy().astype("<f4", copy=False)
        return arr.tobytes()
    if dtype in ("bf16", "bfloat16"):
        # torch.bfloat16 numpy conversion is not stable across environments; store raw bf16 payload.
        t = x.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
        # View as uint16 payload.
        return t.view(torch.uint16).numpy().astype("<u2", copy=False).tobytes()
    raise ValueError(f"Unsupported dtype: {dtype}")


def _parse_layers(state: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> int:
    max_layer = -1
    for k in state.keys():
        for pref in prefixes:
            if not k.startswith(pref + "."):
                continue
            parts = k.split(".")
            if len(parts) < 4:
                continue
            try:
                layer = int(parts[1])
            except ValueError:
                continue
            max_layer = max(max_layer, layer)
    if max_layer < 0:
        raise ValueError("Failed to infer num_layers from checkpoint keys")
    return max_layer + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert KV-compressor .pth to InfiniLM .bin format")
    ap.add_argument("--input", required=True, help="Path to .pth")
    ap.add_argument("--output", required=True, help="Path to output .bin")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Output weight dtype")
    ap.add_argument("--compression-factor", type=int, default=0, help="Header compression_factor (0 = infer from shape)")
    ap.add_argument("--min-seq-len", type=int, default=2, help="Header min_seq_len")
    ap.add_argument(
        "--prefix-order",
        default="compress_tk,compress_tv,compress_ik,compress_iv",
        help="Comma-separated prefixes to export, in per-layer order",
    )
    ap.add_argument(
        "--slot-order",
        default="0,3,6",
        help="Comma-separated slot indices for the 3 MLP linear layers in the .pth",
    )
    ap.add_argument("--num-heads", type=int, default=0, help="Header num_heads (optional)")
    ap.add_argument("--head-dim", type=int, default=0, help="Header head_dim (optional)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    ckpt = torch.load(str(in_path), map_location="cpu")
    state = _extract_compressor_state(ckpt)

    prefixes = [p.strip() for p in str(args.prefix_order).split(",") if p.strip()]
    slots = [int(s.strip()) for s in str(args.slot_order).split(",") if s.strip()]
    if len(slots) != 3:
        raise ValueError("--slot-order must have exactly 3 entries (e.g. 0,3,6)")

    num_layers = _parse_layers(state, prefixes)
    dtype_code, np_dtype = _dtype_code(args.dtype)

    # Infer hidden size / factor / head_dim from the first layer's first weight.
    w0 = state[f"{prefixes[0]}.0.{slots[0]}.weight"]
    if w0.ndim != 2:
        raise ValueError("Expected 2D linear weights")
    out_dim, in_dim = int(w0.shape[0]), int(w0.shape[1])

    head_dim = int(args.head_dim) if int(args.head_dim) > 0 else out_dim
    hidden_size = in_dim
    if hidden_size % head_dim != 0:
        raise ValueError(f"Cannot infer compression factor: hidden_size={hidden_size} head_dim={head_dim}")
    inferred_factor = hidden_size // head_dim
    compression_factor = int(args.compression_factor) if int(args.compression_factor) > 0 else inferred_factor

    weight_count_per_layer = len(prefixes) * len(slots)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        header = struct.pack(
            "<IIHHIIIIIIII",
            MAGIC,
            VERSION,
            dtype_code,
            0,  # reserved
            int(num_layers),
            int(args.num_heads),
            int(head_dim),
            int(hidden_size),
            int(compression_factor),
            int(args.min_seq_len),
            int(weight_count_per_layer),
            0,  # metadata_size_bytes
        )
        f.write(header)

        for layer in range(num_layers):
            for pref in prefixes:
                for slot in slots:
                    w_key = f"{pref}.{layer}.{slot}.weight"
                    b_key = f"{pref}.{layer}.{slot}.bias"
                    if w_key not in state:
                        raise KeyError(f"Missing weight: {w_key}")
                    w = state[w_key]
                    if w.ndim != 2:
                        raise ValueError(f"Expected 2D weight for {w_key}, got {tuple(w.shape)}")
                    rows, cols = int(w.shape[0]), int(w.shape[1])
                    has_bias = 1 if b_key in state and int(state[b_key].numel()) == rows else 0

                    f.write(struct.pack("<III", rows, cols, has_bias))
                    f.write(_to_numpy_bytes(w, args.dtype))
                    if has_bias:
                        b = state[b_key].reshape(rows)
                        f.write(_to_numpy_bytes(b, args.dtype))

    print(
        f"Wrote {out_path} (layers={num_layers}, weights/layer={weight_count_per_layer}, "
        f"dtype={args.dtype}, factor={compression_factor}, hidden_size={hidden_size}, head_dim={head_dim})"
    )


if __name__ == "__main__":
    main()

