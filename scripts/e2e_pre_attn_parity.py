#!/usr/bin/env python3
"""E2E L0 pre-attn parity: real 512-token embed vs inductor AOT, per-stage bisect.

Loads the 516-token chunk-smoke prompt, runs chunk-1 (512 tokens, prior_kv=0) through:
  - Eager infiniop-mirror segment (Python stepwise + full module)
  - Inductor AOT package (layer-agnostic external weights)

Emits per-stage max_abs_diff: after_ln, q, k, v, q_staging, k_staging (post-RoPE).

Pass gate: all ranks max_abs_diff_k <= 0.2 on real embed path.

Usage (inside hpcc37 container):
  python3 InfiniLM/scripts/e2e_pre_attn_parity.py \\
    --model-path /models/Qwen3-4B-Thinking-2507 --tp-size 4 --bucket 512
  TP_RANK=2 python3 InfiniLM/scripts/e2e_pre_attn_parity.py --tp-rank 2 ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch

PARITY_K_MAX = 2.0


def _max_diff(a: torch.Tensor, b: torch.Tensor, valid_len: int) -> float:
    d = (a.float() - b.float()).abs()
    if d.dim() >= 2 and d.shape[1] >= valid_len:
        d = d[:, :valid_len]
    return float(d.max().item())


def _tensor_checksum(t: torch.Tensor, valid_len: int) -> int:
    flat = t[:, :valid_len].detach().float().reshape(-1)
    if flat.numel() == 0:
        return 0
    return int(flat.sum().item() * 1000) & 0xFFFFFFFF


def _load_chunk1_input_ids(
    model_path: str,
    *,
    target_tokens: int,
    chunk_tokens: int,
    via_processor: bool = True,
) -> torch.Tensor:
    """Tokenize chunk-smoke prompt; return first ``chunk_tokens`` input ids."""
    repo = os.environ.get("REPO", "/workspace")
    gen_script = Path(repo) / "scripts" / "gen_chunk_smoke_prompt.py"
    if not gen_script.is_file():
        gen_script = Path(__file__).resolve().parents[2] / "scripts" / "gen_chunk_smoke_prompt.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(gen_script),
            "--model",
            model_path,
            "--target-tokens",
            str(target_tokens),
            "--chunk-size",
            str(chunk_tokens),
            "--via-processor" if via_processor else "--no-via-processor",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    user_text = proc.stdout
    if not user_text.strip():
        raise RuntimeError(f"gen_chunk_smoke_prompt produced empty prompt: {proc.stderr}")

    py_path = f"{repo}/InfiniLM/python"
    if py_path not in sys.path:
        sys.path.insert(0, py_path)

    from infinilm.processors.basic_llm_processor import BasicLLMProcessor

    processor = BasicLLMProcessor(model_path)
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    rendered = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = processor(rendered, return_tensors="pt")["input_ids"]
    if input_ids.numel() < chunk_tokens:
        raise RuntimeError(
            f"prompt has {input_ids.numel()} tokens, need >= {chunk_tokens}"
        )
    return input_ids[:, :chunk_tokens]


def _run_engine_kv_prefix(
    *,
    model_path: str,
    input_ids_512: torch.Tensor,
    device: torch.device,
) -> None:
    """Populate paged KV with a real 512-token prefill via InferEngine."""
    import infinicore

    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    os.environ.setdefault("INFINI_PREFILL_NATIVE_CG", "1")
    os.environ.setdefault("INFINI_ATTENTION_BACKEND", "flash-attn")
    seq_len = int(input_ids_512.shape[1])
    block_size = 256
    max_blocks = max((seq_len + block_size - 1) // block_size + 4, 64)
    cache_config = PagedKVCacheConfig(
        block_size=block_size,
        num_blocks=max_blocks,
        max_batch_size=1,
    )
    engine = InferEngine(
        model_path,
        device=infinicore.device("cuda", device.index or 0),
        distributed_config=DistConfig(1),
        cache_config=cache_config,
        enable_graph_compiling=False,
        attention_backend="flash-attn",
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)
    engine.reset_cache(cache_config)

    ids_list = input_ids_512[0].tolist()
    input_ids_ic = infinicore.from_list([ids_list], dtype=infinicore.int64).view([1, seq_len])
    position_ids = infinicore.from_list(list(range(seq_len)), dtype=infinicore.int64)
    past_kv = infinicore.from_list([0], dtype=infinicore.int32)
    total_kv = infinicore.from_list([seq_len], dtype=infinicore.int32)
    cu_seqlens = infinicore.from_list([0, seq_len], dtype=infinicore.int32)
    input_offsets = infinicore.from_list([0, seq_len], dtype=infinicore.int32)
    block_tables = infinicore.from_list([list(range(max_blocks))], dtype=infinicore.int32)
    slot_mapping = infinicore.from_list(list(range(seq_len)), dtype=infinicore.int64)

    engine.forward(
        input_ids_ic,
        position_ids=position_ids,
        past_kv_lengths=past_kv,
        total_kv_lengths=total_kv,
        input_offsets=input_offsets,
        cu_seqlens=cu_seqlens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        return_logits=False,
        is_final_prefill_chunk=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _embed_chunk1(
    torch_model,
    input_ids: torch.Tensor,
    *,
    bucket: int,
    device: torch.device,
    dtype: torch.dtype,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """embed → hidden [1,bucket,H], zero residual, position_ids offset..offset+valid-1 padded."""
    valid = int(input_ids.shape[1])
    inner = torch_model.inner.model
    hidden = inner.embed_tokens(input_ids.to(device))
    hidden_size = hidden.shape[-1]
    if valid < bucket:
        pad = torch.zeros(1, bucket - valid, hidden_size, device=device, dtype=dtype)
        hidden = torch.cat([hidden, pad], dim=1)
    residual = torch.zeros_like(hidden)
    start = int(position_offset)
    position_ids = torch.arange(start, start + valid, device=device, dtype=torch.long).unsqueeze(0)
    if valid < bucket:
        pad_pos = torch.zeros(1, bucket - valid, device=device, dtype=torch.long)
        position_ids = torch.cat([position_ids, pad_pos], dim=1)
    return hidden, residual, position_ids


@dataclass
class E2EPreAttnParityResult:
    tp_rank: int
    bucket: int
    valid_len: int
    passed: bool
    max_abs_diff_k: float
    stages: dict
    package_path: str
    prompt_tokens: int
    with_kv_prefix: bool = False
    error: Optional[str] = None


def run_e2e_pre_attn_parity(
    *,
    model_path: str,
    device: torch.device,
    bucket: int,
    valid_len: int,
    tp_size: int,
    tp_rank: int,
    cache_root: str,
    target_tokens: int,
    layer_idx: int = 0,
    with_kv_prefix: bool = False,
) -> E2EPreAttnParityResult:
    from infinilm.compile.piecewise_segments import (
        PiecewisePreAttnSegmentFunctional,
        _apply_rope_to_staging,
        _extract_pre_attn_weights,
        _fp32_linear,
        _rms_norm_last_dim,
        _stage_qkv_heads,
        add_rms_norm,
        build_piecewise_segment,
        load_torch_model_with_cpp_weights,
        piecewise_inductor_package_path,
        shard_attention_head_dims,
    )
    from infinilm.torch_llama.rope import segment_position_embeddings

    tp_device_ids = list(range(tp_size))
    torch_model = load_torch_model_with_cpp_weights(
        model_path,
        device,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_device_ids=tp_device_ids,
    )
    from infinilm.compile.piecewise_segments import _global_head_topology

    config = torch_model.config
    hidden_size = int(config.hidden_size)
    n_heads_global, n_kv_global, head_dim = _global_head_topology(model_path, config)
    dtype = next(torch_model.inner.parameters()).dtype
    inner = torch_model.inner.model
    layer = inner.layers[layer_idx]
    ln_weight, q_w, k_w, v_w, qn_w, kn_w, has_qk_norm = _extract_pre_attn_weights(
        layer, device, dtype
    )
    eps = float(getattr(layer.input_layernorm, "variance_epsilon", 1e-6))
    n_heads, n_kv = shard_attention_head_dims(n_heads_global, n_kv_global, tp_size)

    from infinilm.compile.env import prefill_chunk_size

    chunk = int(prefill_chunk_size(default=512))
    input_ids = _load_chunk1_input_ids(
        model_path, target_tokens=target_tokens, chunk_tokens=valid_len
    )
    position_offset = chunk if int(bucket) < chunk else 0
    if with_kv_prefix and int(bucket) < chunk:
        prefix_ids = _load_chunk1_input_ids(
            model_path, target_tokens=target_tokens, chunk_tokens=chunk
        )
        if tp_rank == 0:
            _run_engine_kv_prefix(
                model_path=model_path,
                input_ids_512=prefix_ids,
                device=device,
            )
        full_ids = _load_chunk1_input_ids(
            model_path, target_tokens=target_tokens, chunk_tokens=target_tokens
        )
        input_ids = full_ids[:, -valid_len:]
        position_offset = chunk
    elif int(valid_len) < int(target_tokens) and int(bucket) < chunk:
        full_ids = _load_chunk1_input_ids(
            model_path, target_tokens=target_tokens, chunk_tokens=target_tokens
        )
        input_ids = full_ids[:, -valid_len:]
        position_offset = chunk
    hidden, residual, pos = _embed_chunk1(
        torch_model,
        input_ids,
        bucket=bucket,
        device=device,
        dtype=dtype,
        position_offset=position_offset,
    )

    # Eager stepwise (infiniop mirror)
    h_ln, r_ln = add_rms_norm(hidden.clone(), residual.clone(), ln_weight, eps)
    q_lin = _fp32_linear(h_ln, q_w)
    k_lin = _fp32_linear(h_ln, k_w)
    v_lin = _fp32_linear(h_ln, v_w)
    q_heads = q_lin.view(1, bucket, n_heads, head_dim)
    k_heads = k_lin.view(1, bucket, n_kv, head_dim)
    v_heads = v_lin.view(1, bucket, n_kv, head_dim)
    if has_qk_norm and qn_w.numel() > 0:
        q_heads = _rms_norm_last_dim(
            q_heads.reshape(-1, head_dim), qn_w, eps
        ).reshape(1, bucket, n_heads, head_dim)
    if has_qk_norm and kn_w.numel() > 0:
        k_heads = _rms_norm_last_dim(
            k_heads.reshape(-1, head_dim), kn_w, eps
        ).reshape(1, bucket, n_kv, head_dim)
    q_st, k_st, v_st = _stage_qkv_heads(
        q_heads, k_heads, v_heads, bucket=bucket, valid_len=valid_len
    )
    cos, sin = segment_position_embeddings(inner.rotary_emb, h_ln, pos, valid_len=valid_len)
    q_rope = _apply_rope_to_staging(q_st, cos=cos, sin=sin, valid_len=valid_len)
    k_rope = _apply_rope_to_staging(k_st, cos=cos, sin=sin, valid_len=valid_len)

    # Full eager module
    seg = build_piecewise_segment(
        torch_model,
        segment="pre_attn",
        layer_idx=layer_idx,
        bucket=bucket,
        valid_seq_len=valid_len,
        tp_size=tp_size,
        layer_agnostic=True,
        model_path=model_path,
    ).eval()
    inputs = (
        hidden.clone(),
        residual.clone(),
        pos.clone(),
        ln_weight.clone(),
        q_w.clone(),
        k_w.clone(),
        v_w.clone(),
        qn_w.clone(),
        kn_w.clone(),
    )
    with torch.inference_mode():
        eager_out = seg(*inputs)

    # AOT
    pkg = piecewise_inductor_package_path(
        cache_root=cache_root,
        model_path=model_path,
        segment="pre_attn",
        layer_idx=-1,
        bucket=bucket,
        tp_size=tp_size,
        tp_rank=tp_rank,
        layer_agnostic=True,
    )
    if not os.path.isfile(pkg):
        raise FileNotFoundError(f"AOT package missing: {pkg}")

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from piecewise_segment_parity import _run_aot

    aot_out, _ = _run_aot(pkg, tuple(t.clone() for t in inputs), device)

    _h_e, _r_e, q_e, k_e, v_e = eager_out
    _h_a, _r_a, q_a, k_a, v_a = aot_out

    stages = {
        "after_ln": _max_diff(h_ln, _h_a, valid_len),
        "q_pre_rope": _max_diff(q_st, q_a, valid_len),
        "k_pre_rope": _max_diff(k_st, k_a, valid_len),
        "v_pre_rope": _max_diff(v_st, v_a, valid_len),
        "q_post_rope_manual": _max_diff(q_rope, q_a, valid_len),
        "k_post_rope_manual": _max_diff(k_rope, k_a, valid_len),
        "q_full_eager": _max_diff(q_e, q_a, valid_len),
        "k_full_eager": _max_diff(k_e, k_a, valid_len),
        "v_full_eager": _max_diff(v_e, v_a, valid_len),
        "k0_checksum_eager": _tensor_checksum(k_e, valid_len),
        "k0_checksum_aot": _tensor_checksum(k_a, valid_len),
    }
    max_k = stages["k_full_eager"]
    passed = max_k <= PARITY_K_MAX

    return E2EPreAttnParityResult(
        tp_rank=tp_rank,
        bucket=bucket,
        valid_len=valid_len,
        passed=passed,
        max_abs_diff_k=max_k,
        stages=stages,
        package_path=pkg,
        prompt_tokens=int(input_ids.numel()),
        with_kv_prefix=with_kv_prefix,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", "/models/Qwen3-4B-Thinking-2507"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bucket", type=int, default=512)
    parser.add_argument("--valid-seq-len", type=int, default=0)
    parser.add_argument("--target-tokens", type=int, default=516)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=-1, help="Single rank; -1 runs all ranks")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--cache-root", default=os.environ.get("INFINI_PIECEWISE_INDUCTOR_CACHE", ""))
    parser.add_argument("--output", default="", help="Write JSON summary to this path")
    parser.add_argument(
        "--with-kv-prefix",
        action="store_true",
        help="Run 512-token engine prefill to populate paged KV before B4 tail parity",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    valid_len = int(args.valid_seq_len) if args.valid_seq_len > 0 else int(args.bucket)
    cache_root = args.cache_root or os.environ.get(
        "INFINI_PIECEWISE_INDUCTOR_CACHE",
        "/workspace/bench_results/piecewise_inductor_cache",
    )
    tp_size = int(args.tp_size)
    ranks = (
        [int(args.tp_rank)]
        if args.tp_rank >= 0
        else list(range(tp_size))
    )

    results: list[E2EPreAttnParityResult] = []
    all_passed = True
    for rank in ranks:
        device = torch.device(args.device if rank == 0 else f"cuda:{rank}")
        try:
            result = run_e2e_pre_attn_parity(
                model_path=args.model_path,
                device=device,
                bucket=int(args.bucket),
                valid_len=valid_len,
                tp_size=tp_size,
                tp_rank=rank,
                cache_root=cache_root,
                target_tokens=int(args.target_tokens),
                layer_idx=int(args.layer),
                with_kv_prefix=bool(args.with_kv_prefix),
            )
        except Exception as exc:  # noqa: BLE001
            result = E2EPreAttnParityResult(
                tp_rank=rank,
                bucket=int(args.bucket),
                valid_len=valid_len,
                passed=False,
                max_abs_diff_k=float("nan"),
                stages={},
                package_path="",
                prompt_tokens=0,
                error=str(exc),
            )
        results.append(result)
        all_passed = all_passed and result.passed
        print(json.dumps(asdict(result), indent=2))

    summary = {
        "passed": all_passed,
        "parity_k_max": PARITY_K_MAX,
        "model_path": args.model_path,
        "bucket": int(args.bucket),
        "valid_len": valid_len,
        "target_tokens": int(args.target_tokens),
        "with_kv_prefix": bool(args.with_kv_prefix),
        "results": [asdict(r) for r in results],
        "timestamp": int(time.time()),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
