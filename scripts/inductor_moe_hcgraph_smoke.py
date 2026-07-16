#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""P4 spike: one-layer MoE AOTI inside hcGraph capture/replay (GPU2).

Compares eager ``inductor_moe_`` host ms vs hcGraph replay (device exec when
capture succeeds). Does not attach to GPU3 fair-grid serve.

Deploy cache is read-only; registers existing ``moe_B16/segment.pt2``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


H, E, N = 2048, 160, 512


@dataclass
class MoeHcGraphSpikeResult:
    passed: bool
    bucket: int
    layer_idx: int
    valid_len: int
    has_device_exec: bool
    last_replay_used_device: bool
    replay_device_ok: int
    replay_op_list_fallback: int
    eager_ms_per_iter: float
    replay_ms_per_iter: float
    device_graph_log: str = ""
    error: Optional[str] = None


def _timed_ms(fn, *, warmup: int, iters: int, sync) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) * 1000.0 / max(iters, 1)


def run_moe_hcgraph_spike(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    valid_len: int,
    warmup: int,
    iters: int,
    device_index: int,
    moe_configs: str = "",
    moe_triton_cache: str = "",
) -> MoeHcGraphSpikeResult:
    os.environ["INFINI_GRAPH_STRICT_REPLAY"] = os.environ.get(
        "INFINI_GRAPH_STRICT_REPLAY", "0"
    )
    os.environ["INFINI_PIECEWISE_VALID_LEN"] = str(int(valid_len))
    os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"
    os.environ["INFINI_MOE_ALLOW_JIT"] = "0"
    if moe_configs:
        os.environ["INFINI_MOE_CONFIGS"] = moe_configs
    if moe_triton_cache:
        os.environ["INFINI_MOE_TRITON_CACHE"] = moe_triton_cache
        os.environ["TRITON_CACHE_DIR"] = moe_triton_cache
    if not os.environ.get("INFINI_MOE_CONFIGS", "").strip():
        raise RuntimeError(
            "INFINI_MOE_CONFIGS unset; pass --moe-configs or export it "
            "(deploy cache …/moe_configs)"
        )

    import infinicore
    import torch
    from infinicore.lib import _infinicore as _ic
    from infinilm.compile.piecewise_moe_segment import make_moe_example_inputs
    from infinilm.compile.piecewise_segments import LAYER_AGNOSTIC_IDX, SEGMENT_MOE
    from infinilm.torch_llama.moe_ops import register_fused_moe_routed_op

    register_fused_moe_routed_op()

    pkg = Path(segment_pt2).resolve()
    if not pkg.is_file():
        return MoeHcGraphSpikeResult(
            passed=False,
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=0.0,
            replay_ms_per_iter=0.0,
            error=f"missing segment: {pkg}",
        )

    device = infinicore.device("cuda", int(device_index))
    infinicore.set_device(device)
    cuda_dev = f"cuda:{int(device_index)}"
    dtype = torch.bfloat16

    _ic.register_piecewise_inductor_package(
        SEGMENT_MOE,
        LAYER_AGNOSTIC_IDX,
        int(bucket),
        str(pkg),
        0,
        True,
    )
    if hasattr(_ic, "set_piecewise_inductor_lookup_tp_rank"):
        _ic.set_piecewise_inductor_lookup_tp_rank(0)

    # Random layer-agnostic weights for the resolver (same shapes as export).
    examples = make_moe_example_inputs(
        bucket=bucket,
        hidden_size=H,
        moe_intermediate_size=N,
        n_routed_experts=E,
        device=torch.device(cuda_dev),
        dtype=dtype,
    )
    _hidden_ex, gate_w, bias, w_gu, w_d, shared_gu, shared_d = examples
    _ic.register_moe_external_weights(
        int(layer_idx),
        *[
            infinicore.from_torch(t.contiguous())._underlying
            for t in (gate_w, bias, w_gu, w_d, shared_gu, shared_d)
        ],
    )

    # Decode-like: seq may be 1 while package expects bucket width (C++ pads).
    seq = int(valid_len) if valid_len > 0 else 1
    seq = min(seq, bucket)
    hidden_t = torch.randn(1, seq, H, device=cuda_dev, dtype=dtype)
    out_t = torch.empty(1, seq, H, device=cuda_dev, dtype=dtype)
    hidden = infinicore.from_torch(hidden_t)
    out = infinicore.from_torch(out_t)

    def sync():
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

    def eager_once():
        _ic.inductor_moe_(
            hidden._underlying,
            out._underlying,
            int(layer_idx),
            int(bucket),
        )

    # Warm AOTI / Triton before capture so instantiate does not JIT under stream capture.
    for _ in range(max(warmup, 3)):
        eager_once()
    sync()

    eager_ms = _timed_ms(eager_once, warmup=2, iters=iters, sync=sync)

    # Capture one MoE layer. Triton opaque fused_moe_routed is often not
    # stream-capture-safe; classify known failures for OPT_NOTES.
    capture_error = None
    graph = None
    try:
        infinicore.start_graph_recording(device)
        _ic.inductor_moe_(
            hidden._underlying,
            out._underlying,
            int(layer_idx),
            int(bucket),
        )
        graph = infinicore.stop_graph_recording()
    except Exception as exc:  # noqa: BLE001
        capture_error = str(exc)
        return MoeHcGraphSpikeResult(
            passed=False,
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=eager_ms,
            replay_ms_per_iter=0.0,
            error=(
                f"capture/instantiate failed (likely Triton opaque not "
                f"stream-capture-safe): {capture_error}"
            ),
        )

    has_exec = bool(graph.has_device_exec())
    log_msg = graph.device_graph_log() or ""

    def replay_once():
        graph.run()

    try:
        replay_ms = _timed_ms(replay_once, warmup=2, iters=iters, sync=sync)
        replay_device = bool(graph.last_replay_used_device())
        fallback = int(graph.replay_op_list_fallback())
        device_ok = int(graph.replay_device_ok())
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=has_exec,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=eager_ms,
            replay_ms_per_iter=0.0,
            device_graph_log=log_msg,
            error=f"replay failed: {exc}",
        )

    # Spike "pass" = capture did not throw and replay ran; device exec is best-effort
    # (Triton/opaque may force op-list fallback — still useful vs eager).
    passed = replay_ms > 0 and not (log_msg and "failed" in log_msg.lower() and not has_exec)
    if has_exec and device_ok < 1 and fallback > 0:
        # Device instantiate existed but every launch fell back — still a valid spike.
        passed = True

    return MoeHcGraphSpikeResult(
        passed=passed,
        bucket=bucket,
        layer_idx=layer_idx,
        valid_len=valid_len,
        has_device_exec=has_exec,
        last_replay_used_device=replay_device,
        replay_device_ok=device_ok,
        replay_op_list_fallback=fallback,
        eager_ms_per_iter=eager_ms,
        replay_ms_per_iter=replay_ms,
        device_graph_log=log_msg,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segment-pt2",
        default="",
        help="Path to moe_B16/segment.pt2 (default: deploy cache)",
    )
    parser.add_argument("--bucket", type=int, default=16)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--valid-len", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--moe-configs",
        default="",
        help="INFINI_MOE_CONFIGS dir (default: sibling of segment under deploy cache)",
    )
    parser.add_argument(
        "--moe-triton-cache",
        default="",
        help="INFINI_MOE_TRITON_CACHE dir",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    default_pkg = (
        root
        / "bench_results/piecewise_inductor_cache_minicpm5"
        / "minicpm5.16a3.v0314_057ba9e7b8f5"
        / "tp1/rank0/moe_B16/segment.pt2"
    )
    segment = args.segment_pt2 or str(default_pkg)
    hash_root = Path(segment).resolve().parents[3]  # …/<model_hash> from …/tp1/rank0/moe_B*/segment.pt2
    moe_configs = args.moe_configs or str(hash_root / "moe_configs")
    moe_triton = args.moe_triton_cache or str(hash_root / "moe_triton_cache")

    try:
        result = run_moe_hcgraph_spike(
            segment_pt2=segment,
            bucket=args.bucket,
            layer_idx=args.layer_idx,
            valid_len=args.valid_len if args.valid_len > 0 else 1,
            warmup=args.warmup,
            iters=args.iters,
            device_index=args.device_index,
            moe_configs=moe_configs,
            moe_triton_cache=moe_triton,
        )
    except Exception as exc:  # noqa: BLE001
        result = MoeHcGraphSpikeResult(
            passed=False,
            bucket=args.bucket,
            layer_idx=args.layer_idx,
            valid_len=args.valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=0.0,
            replay_ms_per_iter=0.0,
            error=str(exc),
        )

    status = "PASS" if result.passed else "FAIL"
    speedup = (
        result.eager_ms_per_iter / result.replay_ms_per_iter
        if result.replay_ms_per_iter > 0
        else 0.0
    )
    print(
        f"[moe_hcgraph_spike] {status} bucket={result.bucket} L{result.layer_idx} "
        f"valid={result.valid_len} has_device_exec={result.has_device_exec} "
        f"last_replay_used_device={result.last_replay_used_device} "
        f"replay_device_ok={result.replay_device_ok} "
        f"replay_op_list_fallback={result.replay_op_list_fallback}",
        flush=True,
    )
    print(
        f"[moe_hcgraph_spike] eager_ms/iter={result.eager_ms_per_iter:.3f} "
        f"replay_ms/iter={result.replay_ms_per_iter:.3f} "
        f"eager/replay={speedup:.2f}x",
        flush=True,
    )
    if result.error:
        print(f"[moe_hcgraph_spike] error: {result.error}", flush=True)
    if result.device_graph_log:
        print(
            f"[moe_hcgraph_spike] device_graph_log: {result.device_graph_log[:500]}",
            flush=True,
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

    # Avoid AOT runner teardown double-free masking harness exit code.
    os._exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
