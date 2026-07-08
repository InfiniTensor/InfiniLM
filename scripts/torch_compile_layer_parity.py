#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""PRD-04 M3 triangulation: layer bisect + RoPE / attention isolation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class LayerBisectResult:
    seq_len: int
    first_diverging_layer: int
    layers_compared: int
    max_abs_diff_by_layer: List[float]


@dataclass
class RopeCompareResult:
    positions: List[int]
    max_abs_diff_cos: List[float]
    max_abs_diff_sin: List[float]
    passed: bool


def _load_torch_model(model_path: str, device: torch.device):
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from infinilm.torch_llama.model import load_torch_llama

    import infinicore

    infini_device = infinicore.device("cuda", device.index or 0)
    engine = InferEngine(
        model_path,
        device=infini_device,
        distributed_config=DistConfig(1),
        enable_graph_compiling=False,
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)
    cpp_state_dict = engine._cpp_state_dict_for_compile()
    del engine
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return load_torch_llama(
        model_path,
        device=device,
        splitting_flash_boundary=True,
        cpp_state_dict=cpp_state_dict,
    )


def _forward_hidden_per_layer(
    model,
    input_ids: torch.Tensor,
    *,
    use_longrope: bool = True,
) -> List[torch.Tensor]:
    from infinilm.torch_llama.rope import rotary_embeddings_compile_friendly

    inner = model.inner
    hf_model = inner.model
    bucket_len = int(input_ids.shape[1])
    hidden = hf_model.embed_tokens(input_ids)
    device = hidden.device
    cache_position = torch.arange(0, bucket_len, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)

    if use_longrope:
        position_embeddings = rotary_embeddings_compile_friendly(
            hf_model.rotary_emb, hidden, position_ids
        )
    else:
        out = hf_model.rotary_emb(hidden, position_ids)
        position_embeddings = (out[0], out[1])

    hiddens: List[torch.Tensor] = [hidden.detach().float().cpu()]
    for decoder_layer in hf_model.layers:
        hidden = decoder_layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hiddens.append(hidden.detach().float().cpu())
    return hiddens


def run_layer_bisect(
    model_path: str,
    *,
    seq_len: int,
    device: torch.device,
    seed: int,
    rtol: float,
    atol: float,
) -> LayerBisectResult:
    torch_model = _load_torch_model(model_path, device)
    vocab_size = int(torch_model.config.vocab_size)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + seq_len)
    input_ids = torch.randint(1, vocab_size, (1, seq_len), generator=gen)
    input_ids = input_ids.to(device)

    with torch.inference_mode():
        ref_layers = _forward_hidden_per_layer(torch_model, input_ids, use_longrope=True)
        cmp_layers = _forward_hidden_per_layer(torch_model, input_ids, use_longrope=False)

    diffs = [
        float((ref_layers[i] - cmp_layers[i]).abs().max().item())
        for i in range(min(len(ref_layers), len(cmp_layers)))
    ]
    first_div = len(diffs)
    for i, diff in enumerate(diffs):
        if not torch.allclose(ref_layers[i], cmp_layers[i], rtol=rtol, atol=atol):
            first_div = i
            break

    return LayerBisectResult(
        seq_len=seq_len,
        first_diverging_layer=first_div,
        layers_compared=len(diffs),
        max_abs_diff_by_layer=diffs,
    )


def run_rope_compare(
    model_path: str,
    *,
    positions: Sequence[int],
    device: torch.device,
) -> RopeCompareResult:
    from infinilm.torch_llama.model import load_torch_llama
    from infinilm.torch_llama.rope import rotary_embeddings_compile_friendly

    torch_model = load_torch_llama(model_path, device=device)
    rotary = torch_model.inner.model.rotary_emb
    head_dim = int(torch_model.config.hidden_size // torch_model.config.num_attention_heads)

    max_cos: List[float] = []
    max_sin: List[float] = []
    dummy = torch.zeros(1, 1, head_dim, device=device, dtype=torch.bfloat16)
    for pos in positions:
        position_ids = torch.tensor([[pos]], device=device, dtype=torch.long)
        cos_t, sin_t = rotary_embeddings_compile_friendly(rotary, dummy, position_ids)
        cos_t = cos_t[0, 0].float().cpu()
        sin_t = sin_t[0, 0].float().cpu()

        with torch.inference_mode():
            ref = rotary(dummy, position_ids)
            cos_ref = ref[0][0, 0].float().cpu()
            sin_ref = ref[1][0, 0].float().cpu()

        max_cos.append(float((cos_t - cos_ref).abs().max().item()))
        max_sin.append(float((sin_t - sin_ref).abs().max().item()))

    passed = all(c < 1e-3 and s < 1e-3 for c, s in zip(max_cos, max_sin))
    return RopeCompareResult(
        positions=list(positions),
        max_abs_diff_cos=max_cos,
        max_abs_diff_sin=max_sin,
        passed=passed,
    )


def run_attn_flash_vs_matmul(
    model_path: str,
    *,
    seq_len: int,
    device: torch.device,
    seed: int,
) -> dict:
    from infinilm.torch_llama.attention import (
        register_splitting_flash_attention,
        splitting_flash_attention_forward,
    )
    from transformers.models.llama.modeling_llama import eager_attention_forward

    torch_model = _load_torch_model(model_path, device)
    layer0 = torch_model.inner.model.layers[0].self_attn
    hidden_size = int(torch_model.config.hidden_size)
    n_heads = int(torch_model.config.num_attention_heads)
    n_kv = int(torch_model.config.num_key_value_heads)
    head_dim = hidden_size // n_heads

    gen = torch.Generator(device=device)
    gen.manual_seed(seed + seq_len + 17)
    q = torch.randn(1, n_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(1, n_kv, seq_len, head_dim, device=device, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(1, n_kv, seq_len, head_dim, device=device, dtype=torch.bfloat16, generator=gen)

    register_splitting_flash_attention()
    with torch.inference_mode():
        flash_out, _ = splitting_flash_attention_forward(
            layer0, q, k, v, None, scaling=layer0.scaling, is_causal=True
        )
        matmul_out, _ = eager_attention_forward(
            layer0, q, k, v, None, scaling=layer0.scaling, is_causal=True
        )

    diff = (flash_out.float() - matmul_out.float()).abs()
    return {
        "seq_len": seq_len,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "token_match": bool(flash_out.argmax(-1).equal(matmul_out.argmax(-1))),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-lens", default="512,4096")
    parser.add_argument(
        "--mode",
        choices=("layer_bisect", "rope", "attn"),
        default="layer_bisect",
    )
    parser.add_argument(
        "--rope-positions",
        default="0,511,4095",
        help="positions for --mode rope",
    )
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    summary: dict = {"mode": args.mode, "model_path": args.model_path}

    if args.mode == "layer_bisect":
        results = [
            run_layer_bisect(
                args.model_path,
                seq_len=sl,
                device=device,
                seed=args.seed,
                rtol=args.rtol,
                atol=args.atol,
            )
            for sl in seq_lens
        ]
        summary["results"] = [asdict(r) for r in results]
        for r in results:
            print(
                f"[triangulate] layer_bisect seq_len={r.seq_len} "
                f"first_diverging_layer={r.first_diverging_layer} "
                f"layers={r.layers_compared}"
            )
    elif args.mode == "rope":
        positions = [int(x.strip()) for x in args.rope_positions.split(",") if x.strip()]
        result = run_rope_compare(args.model_path, positions=positions, device=device)
        summary["rope"] = asdict(result)
        print(
            f"[triangulate] rope_compare passed={result.passed} "
            f"cos_diff={result.max_abs_diff_cos} sin_diff={result.max_abs_diff_sin}"
        )
    else:
        attn_results = [
            run_attn_flash_vs_matmul(
                args.model_path, seq_len=sl, device=device, seed=args.seed
            )
            for sl in seq_lens
        ]
        summary["attn"] = attn_results
        for row in attn_results:
            print(
                f"[triangulate] attn seq_len={row['seq_len']} "
                f"max_abs_diff={row['max_abs_diff']:.6f} "
                f"token_match={row['token_match']}"
            )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
