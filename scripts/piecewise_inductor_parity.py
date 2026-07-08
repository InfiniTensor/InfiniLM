#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""M4 Phase 3: native piecewise prefill parity — inductor segments vs infiniop baseline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import torch

PARITY_MAX_ABS_DIFF = 0.2


@dataclass
class PrefillParityResult:
    seq_len: int
    passed: bool
    token_match: bool
    max_abs_diff: float
    mean_abs_diff: float
    baseline_argmax: int
    inductor_argmax: int
    baseline_ms: float
    inductor_ms: float
    registered_packages: int = 0
    error: Optional[str] = None


def _make_input_ids(seq_len: int, vocab_size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + seq_len)
    return torch.randint(1, vocab_size, (1, seq_len), generator=gen, dtype=torch.long)


def _run_native_piecewise_prefill(
    *,
    model_path: str,
    input_ids: torch.Tensor,
    inductor_segment: bool,
    compile_segments: bool,
    cache_root: str,
    num_layers: int,
) -> tuple[torch.Tensor, float, int]:
    import infinicore

    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    seq_len = int(input_ids.shape[1])
    os.environ["INFINI_PREFILL_NATIVE_CG"] = "1"
    attn = os.environ.get("INFINI_ATTENTION_BACKEND", "flash-attn")
    if attn == "flash":
        attn = "flash-attn"
    os.environ["INFINI_ATTENTION_BACKEND"] = attn
    os.environ["INFINI_RETURN_LOGITS"] = "1"
    os.environ.pop("INFINI_TORCH_COMPILE", None)
    if inductor_segment:
        os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"
    else:
        os.environ.pop("INFINI_PIECEWISE_INDUCTOR_SEGMENT", None)

    if compile_segments and inductor_segment:
        from infinilm.compile.piecewise_segments import (
            SEGMENT_PRE_ATTN,
            aot_compile_piecewise_segments_batch,
        )

        device = torch.device("cuda:0")
        aot_compile_piecewise_segments_batch(
            model_path=model_path,
            segment=SEGMENT_PRE_ATTN,
            layer_indices=range(num_layers),
            bucket=seq_len,
            device=device,
            cache_root=cache_root,
            valid_seq_len=seq_len,
        )

    block_size = 16
    max_blocks = (seq_len + block_size - 1) // block_size + 4
    cache_config = PagedKVCacheConfig(
        block_size=block_size,
        num_blocks=max(max_blocks * 2, 64),
        max_batch_size=1,
    )

    engine = InferEngine(
        model_path,
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(1),
        cache_config=cache_config,
        enable_graph_compiling=False,
        attention_backend=attn,
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)
    engine.reset_cache(cache_config)

    registered = 0
    if inductor_segment:
        try:
            from infinicore.compiled_subgraphs import register_piecewise_inductor_packages
            from infinilm.compile.piecewise_segments import SEGMENT_PRE_ATTN

            registered = register_piecewise_inductor_packages(
                model_path=model_path,
                segments=(SEGMENT_PRE_ATTN,),
                layer_indices=range(num_layers),
                buckets=(seq_len,),
                cache_root=cache_root,
            )
        except Exception:
            registered = 0

    ids_list = input_ids[0].tolist()
    # Paged prefill: flatten batch into seq dim (matches InferEngine.generate iter 0).
    input_ids_ic = infinicore.from_list([ids_list], dtype=infinicore.int64).view([1, seq_len])
    position_ids = infinicore.from_list(list(range(seq_len)), dtype=infinicore.int64)
    past_kv = infinicore.from_list([0], dtype=infinicore.int32)
    total_kv = infinicore.from_list([seq_len], dtype=infinicore.int32)
    cu_seqlens = infinicore.from_list([0, seq_len], dtype=infinicore.int32)
    input_offsets = infinicore.from_list([0, seq_len], dtype=infinicore.int32)
    block_tables = infinicore.from_list([list(range(max_blocks))], dtype=infinicore.int32)
    slot_mapping = infinicore.from_list(list(range(seq_len)), dtype=infinicore.int64)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.forward(
        input_ids_ic,
        position_ids=position_ids,
        past_kv_lengths=past_kv,
        total_kv_lengths=total_kv,
        input_offsets=input_offsets,
        cu_seqlens=cu_seqlens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        return_logits=True,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        is_final_prefill_chunk=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    last = infinicore.to_torch(out).float().cpu()[0, seq_len - 1, :].clone()
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return last, ms, registered


def run_prefill_parity(
    *,
    model_path: str,
    seq_len: int,
    seed: int,
    cache_root: str,
    compile_segments: bool,
    num_layers: Optional[int] = None,
) -> PrefillParityResult:
    import json as _json

    if num_layers is None:
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            num_layers = int(_json.load(f).get("num_hidden_layers", 0))

    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        vocab_size = int(_json.load(f).get("vocab_size", 73448))

    input_ids = _make_input_ids(seq_len, vocab_size, seed)
    try:
        baseline, baseline_ms, _ = _run_native_piecewise_prefill(
            model_path=model_path,
            input_ids=input_ids,
            inductor_segment=False,
            compile_segments=False,
            cache_root=cache_root,
            num_layers=num_layers,
        )
        inductor, inductor_ms, registered = _run_native_piecewise_prefill(
            model_path=model_path,
            input_ids=input_ids,
            inductor_segment=True,
            compile_segments=compile_segments,
            cache_root=cache_root,
            num_layers=num_layers,
        )
        diff = (baseline - inductor).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        baseline_argmax = int(baseline.argmax().item())
        inductor_argmax = int(inductor.argmax().item())
        token_match = baseline_argmax == inductor_argmax
        passed = max_abs <= PARITY_MAX_ABS_DIFF and token_match
        return PrefillParityResult(
            seq_len=seq_len,
            passed=passed,
            token_match=token_match,
            max_abs_diff=max_abs,
            mean_abs_diff=mean_abs,
            baseline_argmax=baseline_argmax,
            inductor_argmax=inductor_argmax,
            baseline_ms=baseline_ms,
            inductor_ms=inductor_ms,
            registered_packages=registered,
        )
    except Exception as exc:  # noqa: BLE001
        return PrefillParityResult(
            seq_len=seq_len,
            passed=False,
            token_match=False,
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            baseline_argmax=-1,
            inductor_argmax=-1,
            baseline_ms=0.0,
            inductor_ms=0.0,
            error=str(exc),
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--seq-lens", default="512,4096")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-root", default="")
    parser.add_argument(
        "--compile-segments",
        action="store_true",
        help="AOT-compile missing pre_attn packages before inductor run",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from infinilm.compile.env import piecewise_inductor_cache_root

    cache_root = args.cache_root or piecewise_inductor_cache_root()
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]

    results: List[PrefillParityResult] = []
    exit_code = 0
    for seq_len in seq_lens:
        print(
            f"[piecewise_inductor_parity] seq_len={seq_len} model={args.model_path}",
            flush=True,
        )
        result = run_prefill_parity(
            model_path=args.model_path,
            seq_len=seq_len,
            seed=args.seed,
            cache_root=cache_root,
            compile_segments=args.compile_segments,
        )
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[piecewise_inductor_parity] {status} seq_len={seq_len} "
            f"max_abs_diff={result.max_abs_diff:.6f} token_match={result.token_match} "
            f"baseline_argmax={result.baseline_argmax} inductor_argmax={result.inductor_argmax} "
            f"registered={result.registered_packages} "
            f"baseline_ms={result.baseline_ms:.2f} inductor_ms={result.inductor_ms:.2f}",
            flush=True,
        )
        if result.error:
            print(f"[piecewise_inductor_parity] error: {result.error}", flush=True)
        if not result.passed:
            exit_code = 1

    summary = {
        "model_path": args.model_path,
        "parity_max_abs_diff_gate": PARITY_MAX_ABS_DIFF,
        "results": [asdict(r) for r in results],
        "passed": exit_code == 0,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
