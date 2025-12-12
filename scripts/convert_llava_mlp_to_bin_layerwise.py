#!/usr/bin/env python3
import argparse
import struct
import torch

PREFIX_ORDER = ["compress_tk", "compress_tv", "compress_ik", "compress_iv"]
WEIGHTS_PER_PREFIX = 3  # slots 0,1,2


def parse_args():
    ap = argparse.ArgumentParser(description="Convert LLaVA KV compressor weights to bin (layer-wise layout).")
    ap.add_argument("--pth", required=True, help="Path to llava_mlp.pth")
    ap.add_argument("--out", required=True, help="Output bin path")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--compression-factor", type=int, required=True)
    ap.add_argument("--min-seq-len", type=int, default=1)
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


def group_keys(flat: dict):
    """
    Expect keys like <prefix>.<layer>.<slot>.weight/bias
    """
    grouped = []
    for key, tensor in flat.items():
        parts = key.split(".")
        if len(parts) < 4:
            continue
        prefix, layer_str, slot_str, kind = parts[0], parts[1], parts[2], parts[3]
        if prefix not in PREFIX_ORDER or kind not in ("weight", "bias"):
            continue
        try:
            layer = int(layer_str)
            slot = int(slot_str)
        except ValueError:
            continue
        grouped.append((prefix, layer, slot, kind == "bias", tensor))
    return grouped


def write_header(f, num_layers, num_heads, head_dim, hidden_size,
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
    f.write(t.cpu().contiguous().numpy().tobytes())


def main():
    args = parse_args()
    state = torch.load(args.pth, map_location="cpu")
    model_state_dict = state.get("model_state_dict", state)
    flat = collect_tensors(model_state_dict)

    # Focus on compressor subtree if present.
    compressor_keys = {k: v for k, v in flat.items() if k.startswith("compressor.")}
    if compressor_keys:
        flat = {k.split("compressor.", 1)[1]: v for k, v in compressor_keys.items()}

    grouped = group_keys(flat)
    if not grouped:
        raise RuntimeError("No compatible weight/bias keys found (expected prefix in "
                           f"{PREFIX_ORDER} and pattern <prefix>.<layer>.<slot>.weight)")

    num_layers = max(g[1] for g in grouped) + 1

    sample = next(t for (_, _, _, is_bias, t) in grouped if not is_bias)
    if sample.ndim < 2:
        raise RuntimeError(f"Sample tensor is not 2D: shape={sample.shape}")
    hidden_size = sample.shape[-1]

    num_heads = args.num_heads
    head_dim = args.head_dim

    # Build per-layer array: 12 slots (tk/tv/ik/iv each 3 slots)
    per_layer = [[{"weight": None, "bias": None} for _ in range(len(PREFIX_ORDER) * WEIGHTS_PER_PREFIX)]
                 for _ in range(num_layers)]
    for (prefix, layer, slot, is_bias, tensor) in grouped:
        if prefix not in PREFIX_ORDER or slot >= WEIGHTS_PER_PREFIX or layer >= num_layers:
            continue
        pidx = PREFIX_ORDER.index(prefix)
        idx = pidx * WEIGHTS_PER_PREFIX + slot
        per_layer[layer][idx]["bias" if is_bias else "weight"] = tensor

    weight_count_per_layer = len(PREFIX_ORDER) * WEIGHTS_PER_PREFIX  # 12
    with open(args.out, "wb") as f:
        write_header(
            f,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            dtype=args.dtype,
            compression_factor=args.compression_factor,
            min_seq_len=args.min_seq_len,
            weight_count=weight_count_per_layer,
        )

        # Layer-wise layout: for each layer, write tk(3), tv(3), ik(3), iv(3)
        for layer in range(num_layers):
            for idx in range(weight_count_per_layer):
                entry = per_layer[layer][idx]
                if entry["weight"] is None:
                    pidx = idx // WEIGHTS_PER_PREFIX
                    slot = idx % WEIGHTS_PER_PREFIX
                    raise RuntimeError(f"Missing weight for layer {layer} {PREFIX_ORDER[pidx]}.{slot}")
                weight = tensor_to_dtype(entry["weight"], args.dtype)
                rows, cols = weight.shape
                bias_tensor = entry["bias"]
                has_bias = 1 if bias_tensor is not None else 0
                meta = struct.pack("<I I I", rows, cols, has_bias)
                f.write(meta)
                write_tensor(f, weight)
                if has_bias:
                    bias = tensor_to_dtype(bias_tensor, args.dtype)
                    if bias.numel() != rows:
                        pidx = idx // WEIGHTS_PER_PREFIX
                        slot = idx % WEIGHTS_PER_PREFIX
                        raise RuntimeError(
                            f"Bias size mismatch for layer {layer} {PREFIX_ORDER[pidx]}.{slot}: expected {rows}, got {bias.numel()}"
                        )
                    write_tensor(f, bias)

    print(f"Wrote layer-wise bin to {args.out} (layers={num_layers}, weight_count_per_layer={weight_count_per_layer})")


if __name__ == "__main__":
    main()
