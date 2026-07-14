#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Phase 1: hcGraph replay smoke for InductorSegment pre_attn L0 @ B4."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class HcGraphSmokeResult:
    passed: bool
    bucket: int
    layer_idx: int
    has_device_exec: bool
    last_replay_used_device: bool
    replay_device_ok: int
    replay_op_list_fallback: int
    registered_packages: int
    device_graph_log: str = ""
    error: Optional[str] = None


def _load_model_shapes(model_path: str, *, tp_size: int = 1) -> dict:
    from infinilm.compile.piecewise_segments import shard_attention_head_dims

    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    hidden_size = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    n_heads, n_kv = shard_attention_head_dims(n_heads, n_kv, tp_size)
    head_dim = int(cfg.get("head_dim", hidden_size // int(cfg["num_attention_heads"])))
    return {
        "hidden_size": hidden_size,
        "num_heads": n_heads,
        "num_kv_heads": n_kv,
        "head_dim": head_dim,
        "dtype_name": cfg.get("torch_dtype", "bfloat16"),
        "tp_size": max(1, int(tp_size)),
    }


def run_hcgraph_smoke(
    *,
    model_path: str,
    bucket: int,
    layer_idx: int,
    cache_root: str,
    compile_segments: bool,
    valid_len: Optional[int] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    device_index: int = 0,
) -> HcGraphSmokeResult:
    os.environ["INFINI_GRAPH_STRICT_REPLAY"] = "1"
    os.environ["INFINI_PIECEWISE_VALID_LEN"] = str(int(valid_len) if valid_len is not None else bucket)

    import infinicore
    import torch
    from infinicore.lib import _infinicore as _ic

    shapes = _load_model_shapes(model_path, tp_size=tp_size)
    valid = int(valid_len) if valid_len is not None else bucket
    dtype = torch.bfloat16 if shapes["dtype_name"] in ("bfloat16", "bf16") else torch.float16
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    cuda_dev = f"cuda:{int(device_index)}"
    device = infinicore.device("cuda", int(device_index))

    if compile_segments:
        from infinilm.compile.piecewise_segments import (
            SEGMENT_PRE_ATTN,
            aot_compile_piecewise_segments_batch,
        )

        aot_compile_piecewise_segments_batch(
            model_path=model_path,
            segment=SEGMENT_PRE_ATTN,
            layer_indices=(layer_idx,),
            bucket=bucket,
            device=torch.device(cuda_dev),
            cache_root=cache_root,
            valid_seq_len=valid,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_device_ids=list(range(tp_size)),
        )

    device = infinicore.device("cuda", int(device_index))
    infinicore.set_device(device)

    from infinilm.compile.piecewise_segments import piecewise_layer_agnostic_enabled

    layer_agnostic = piecewise_layer_agnostic_enabled()

    if layer_agnostic:
        from infinilm.compile.piecewise_segments import (
            _extract_pre_attn_weights,
            load_torch_model_with_cpp_weights,
        )

        os.environ["INFINI_PIECEWISE_INDUCTOR_COMPILE_ON_MISS"] = "0"
        try:
            tp_dev_ids = list(range(tp_size))
            torch_model = load_torch_model_with_cpp_weights(
                model_path,
                torch.device(cuda_dev),
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_device_ids=tp_dev_ids,
            )
            inner = torch_model.inner.model
            ln_w, q_w, k_w, v_w, qn_w, kn_w, _has_qk = _extract_pre_attn_weights(
                inner.layers[int(layer_idx)],
                torch.device(cuda_dev),
                dtype,
            )
            ln_w, q_w, k_w, v_w, qn_w, kn_w = (
                t.clone() for t in (ln_w, q_w, k_w, v_w, qn_w, kn_w)
            )
            del torch_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_device(int(device_index))
            infinicore.set_device(device)
            _ic.register_pre_attn_external_weights(
                int(layer_idx),
                *[
                    infinicore.from_torch(t)._underlying
                    for t in (ln_w, q_w, k_w, v_w, qn_w, kn_w)
                ],
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    infinicore.set_device(device)

    from infinilm.compile.piecewise_segments import (
        LAYER_AGNOSTIC_IDX,
        SEGMENT_PRE_ATTN,
        piecewise_inductor_package_path,
    )

    os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"

    registered = 0
    if layer_agnostic:
        package_path = piecewise_inductor_package_path(
            cache_root=cache_root,
            model_path=model_path,
            segment=SEGMENT_PRE_ATTN,
            layer_idx=LAYER_AGNOSTIC_IDX,
            bucket=int(bucket),
            tp_size=tp_size,
            tp_rank=int(tp_rank),
            layer_agnostic=True,
        )
        if os.path.isfile(package_path):
            _ic.register_piecewise_inductor_package(
                SEGMENT_PRE_ATTN,
                LAYER_AGNOSTIC_IDX,
                int(bucket),
                os.path.abspath(package_path),
                int(tp_rank),
                True,
            )
            registered = 1

    if registered == 0:
        return HcGraphSmokeResult(
            passed=False,
            bucket=bucket,
            layer_idx=layer_idx,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            registered_packages=0,
            error="no AOT package registered",
        )

    ic_dtype = (
        infinicore.bfloat16 if dtype == torch.bfloat16 else infinicore.float16
    )
    gen = torch.Generator(device=cuda_dev)
    gen.manual_seed(42 + int(layer_idx) + int(bucket))
    hidden = infinicore.from_torch(
        torch.randn(
            1, bucket, shapes["hidden_size"], device=cuda_dev, dtype=dtype, generator=gen
        )
    )
    residual = infinicore.from_torch(
        torch.randn(
            1, bucket, shapes["hidden_size"], device=cuda_dev, dtype=dtype, generator=gen
        )
    )
    position_offset = 512 if int(valid) < 512 else 0
    positions = infinicore.from_torch(
        torch.arange(
            position_offset, position_offset + valid, device=cuda_dev, dtype=torch.int64
        )
    )
    positions_padded = infinicore.from_torch(
        torch.zeros(1, bucket, device=cuda_dev, dtype=torch.int64)
    )
    q_rope = infinicore.from_torch(
        torch.zeros(
            1, bucket, shapes["num_heads"], shapes["head_dim"],
            device=cuda_dev, dtype=dtype,
        )
    )
    k_rope = infinicore.from_torch(
        torch.zeros(
            1, bucket, shapes["num_kv_heads"], shapes["head_dim"],
            device=cuda_dev, dtype=dtype,
        )
    )
    v_rope = infinicore.from_torch(
        torch.zeros(
            1, bucket, shapes["num_kv_heads"], shapes["head_dim"],
            device=cuda_dev, dtype=dtype,
        )
    )

    if hasattr(_ic, "set_piecewise_inductor_lookup_tp_rank"):
        _ic.set_piecewise_inductor_lookup_tp_rank(int(tp_rank))

    if hasattr(_ic, "warmup_piecewise_inductor_pre_attn"):
        _ic.warmup_piecewise_inductor_pre_attn(
            positions._underlying,
            positions_padded._underlying,
            hidden._underlying,
            residual._underlying,
            int(layer_idx),
            int(bucket),
            int(valid),
        )
        infinicore.sync_stream()

    infinicore.start_graph_recording(device)
    _ic.inductor_segment_(
        positions._underlying,
        hidden._underlying,
        residual._underlying,
        q_rope._underlying,
        k_rope._underlying,
        v_rope._underlying,
        SEGMENT_PRE_ATTN,
        int(layer_idx),
        int(bucket),
    )
    graph = infinicore.stop_graph_recording()

    has_exec = graph.has_device_exec()
    log_msg = graph.device_graph_log()

    graph.run()
    replay1_device = graph.last_replay_used_device()
    fallback1 = graph.replay_op_list_fallback()
    graph.run()
    replay2_device = graph.last_replay_used_device()
    fallback2 = graph.replay_op_list_fallback()
    device_ok = graph.replay_device_ok()

    passed = (
        has_exec
        and replay2_device
        and fallback2 == 0
        and device_ok >= 2
    )

    return HcGraphSmokeResult(
        passed=passed,
        bucket=bucket,
        layer_idx=layer_idx,
        has_device_exec=has_exec,
        last_replay_used_device=replay2_device,
        replay_device_ok=device_ok,
        replay_op_list_fallback=fallback2,
        registered_packages=registered,
        device_graph_log=log_msg,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--bucket", type=int, default=512)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--valid-len", type=int, default=0)
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--compile-segments", action="store_true")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from infinilm.compile.env import piecewise_inductor_cache_root

    cache_root = args.cache_root or piecewise_inductor_cache_root()
    valid_len = args.valid_len if args.valid_len > 0 else args.bucket

    try:
        result = run_hcgraph_smoke(
            model_path=args.model_path,
            bucket=args.bucket,
            layer_idx=args.layer_idx,
            cache_root=cache_root,
            compile_segments=args.compile_segments,
            valid_len=valid_len,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            device_index=args.device_index,
        )
    except Exception as exc:  # noqa: BLE001
        result = HcGraphSmokeResult(
            passed=False,
            bucket=args.bucket,
            layer_idx=args.layer_idx,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            registered_packages=0,
            error=str(exc),
        )

    status = "PASS" if result.passed else "FAIL"
    print(
        f"[hcgraph_smoke] {status} bucket={result.bucket} L{result.layer_idx} "
        f"has_device_exec={result.has_device_exec} "
        f"last_replay_used_device={result.last_replay_used_device} "
        f"replay_device_ok={result.replay_device_ok} "
        f"replay_op_list_fallback={result.replay_op_list_fallback} "
        f"registered={result.registered_packages}",
        flush=True,
    )
    if result.error:
        print(f"[hcgraph_smoke] error: {result.error}", flush=True)
    if result.device_graph_log:
        print(f"[hcgraph_smoke] device_graph_log: {result.device_graph_log}", flush=True)

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

    # Avoid AOT runner teardown double-free masking harness exit code.
    os._exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
