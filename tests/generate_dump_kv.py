#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch


def _extract_state(pth_obj) -> Dict[str, torch.Tensor]:
    if not isinstance(pth_obj, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(pth_obj)}")
    if "model_state_dict" in pth_obj and isinstance(pth_obj["model_state_dict"], dict):
        msd = pth_obj["model_state_dict"]
        if "compressor" in msd and isinstance(msd["compressor"], dict):
            return msd["compressor"]
    if "compressor" in pth_obj and isinstance(pth_obj["compressor"], dict):
        return pth_obj["compressor"]
    raise ValueError("Cannot find compressor state dict in checkpoint")


def _mlp_forward(x2d: torch.Tensor, w: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], b: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    w0, w1, w2 = w
    b0, b1, b2 = b
    # Emulate InfiniLM pipeline: fp16 weights + fp16 outputs after each linear.
    y = (x2d @ w0.t()) + b0
    y = y.to(torch.float16).to(torch.float32)
    y = torch.relu(y)
    y = y.to(torch.float16).to(torch.float32)

    y = (y @ w1.t()) + b1
    y = y.to(torch.float16).to(torch.float32)
    y = torch.relu(y)
    y = y.to(torch.float16).to(torch.float32)

    y = (y @ w2.t()) + b2
    y = y.to(torch.float16)
    return y


def _compress_segment(
    k: torch.Tensor,
    v: torch.Tensor,
    state: Dict[str, torch.Tensor],
    layer: int,
    prefix_k: str,
    prefix_v: str,
    factor: int,
    min_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # k/v: [S, H, D] fp16
    s, h, d = k.shape
    groups = s // factor
    if groups < min_seq_len:
        return k, v
    compress_len = groups * factor
    remainder = s - compress_len

    def _load_mlp(prefix: str):
        # Slots 0/3/6 correspond to linear layers in Sequential.
        slots = [0, 3, 6]
        ws = []
        bs = []
        for slot in slots:
            w = state[f"{prefix}.{layer}.{slot}.weight"].to(torch.float16).to(torch.float32)
            b = state[f"{prefix}.{layer}.{slot}.bias"].to(torch.float16).to(torch.float32)
            ws.append(w)
            bs.append(b)
        return (ws[0], ws[1], ws[2]), (bs[0], bs[1], bs[2])

    w_k, b_k = _load_mlp(prefix_k)
    w_v, b_v = _load_mlp(prefix_v)

    # Head-major: [H, S, D] then group S*D into [H, groups, factor*D].
    k_head = k[:compress_len].permute(1, 0, 2).contiguous()
    v_head = v[:compress_len].permute(1, 0, 2).contiguous()

    k_grouped = k_head.reshape(h, groups, factor * d).reshape(h * groups, factor * d).to(torch.float32)
    v_grouped = v_head.reshape(h, groups, factor * d).reshape(h * groups, factor * d).to(torch.float32)

    k_out2d = _mlp_forward(k_grouped, w_k, b_k)
    v_out2d = _mlp_forward(v_grouped, w_v, b_v)

    k_comp = k_out2d.reshape(h, groups, d).permute(1, 0, 2).contiguous()
    v_comp = v_out2d.reshape(h, groups, d).permute(1, 0, 2).contiguous()

    if remainder == 0:
        return k_comp, v_comp

    k_tail = k[compress_len:].contiguous()
    v_tail = v[compress_len:].contiguous()
    k_cat = torch.cat([k_comp, k_tail], dim=0).contiguous()
    v_cat = torch.cat([v_comp, v_tail], dim=0).contiguous()
    return k_cat, v_cat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", required=True, help="Path to compressor checkpoint (.pth)")
    ap.add_argument("--out-dir", default="dump_kv", help="Output folder (relative or absolute)")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq-in", type=int, default=21)
    ap.add_argument("--image-kv-len", type=int, default=11)
    ap.add_argument("--factor", type=int, default=5)
    ap.add_argument("--min-seq-len", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))

    ckpt = torch.load(args.pth, map_location="cpu")
    state = _extract_state(ckpt)

    layers = int(args.layers)
    heads = int(args.heads)
    d = int(args.head_dim)
    seq_in = int(args.seq_in)
    image_kv_len = int(args.image_kv_len)
    factor = int(args.factor)
    min_seq_len = int(args.min_seq_len)

    if not (0 < image_kv_len < seq_in):
        raise ValueError("--image-kv-len must be in (0, seq_in)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic KV in seq-major layout: [S, H, D] fp16.
    # Keep values small to avoid fp16 overflow in matmul.
    k_in_all = torch.randn(layers, seq_in, heads, d, dtype=torch.float32) * 0.02
    v_in_all = torch.randn(layers, seq_in, heads, d, dtype=torch.float32) * 0.02
    k_in_all = k_in_all.to(torch.float16).to(torch.float32).to(torch.float16)
    v_in_all = v_in_all.to(torch.float16).to(torch.float32).to(torch.float16)

    # Compress per layer.
    k_out_layers = []
    v_out_layers = []
    for layer in range(layers):
        k = k_in_all[layer]
        v = v_in_all[layer]
        k_img = k[:image_kv_len]
        v_img = v[:image_kv_len]
        k_txt = k[image_kv_len:]
        v_txt = v[image_kv_len:]

        k_img_c, v_img_c = _compress_segment(k_img, v_img, state, layer, "compress_ik", "compress_iv", factor, min_seq_len)
        k_txt_c, v_txt_c = _compress_segment(k_txt, v_txt, state, layer, "compress_tk", "compress_tv", factor, min_seq_len)

        k_out = torch.cat([k_img_c, k_txt_c], dim=0).contiguous()
        v_out = torch.cat([v_img_c, v_txt_c], dim=0).contiguous()
        k_out_layers.append(k_out)
        v_out_layers.append(v_out)

    seq_out = int(k_out_layers[0].shape[0])
    if any(int(x.shape[0]) != seq_out for x in k_out_layers + v_out_layers):
        raise RuntimeError("Inconsistent seq_out across layers")

    meta = {
        "layers": layers,
        "heads": heads,
        "head_dim": d,
        "seq_len_in": seq_in,
        "seq_len_out": seq_out,
        "compression_factor": factor,
        "min_seq_len": min_seq_len,
        "it_len": [image_kv_len, 0],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # input_kv.bin layout for tests: [B=1, H, S, D] row-major, per layer K then V.
    input_chunks = []
    for layer in range(layers):
        k_hsd = k_in_all[layer].permute(1, 0, 2).contiguous()  # [H, S, D]
        v_hsd = v_in_all[layer].permute(1, 0, 2).contiguous()
        input_chunks.append(k_hsd)
        input_chunks.append(v_hsd)
    input_blob = (
        torch.cat([t.reshape(-1) for t in input_chunks], dim=0)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .tobytes()
    )
    (out_dir / "input_kv.bin").write_bytes(input_blob)

    # output_kv.bin layout for tests: contiguous [S, H, D] row-major, per layer K then V.
    out_chunks = []
    for layer in range(layers):
        out_chunks.append(k_out_layers[layer].reshape(-1))
        out_chunks.append(v_out_layers[layer].reshape(-1))
    out_blob = (
        torch.cat(out_chunks, dim=0)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .tobytes()
    )
    (out_dir / "output_kv.bin").write_bytes(out_blob)

    print(f"Wrote {out_dir}/meta.json, input_kv.bin, output_kv.bin (seq_out={seq_out})")


if __name__ == "__main__":
    main()
