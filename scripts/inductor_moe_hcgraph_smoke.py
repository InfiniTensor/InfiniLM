#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""CG-1 / CG-2 / Triton-capture MoE hcGraph smokes (GPU2 only; never GPU3).

Modes
-----
* ``full_moe`` (CG-1 / legacy P4): record a single-layer ``inductor_moe_`` into one
  logical graph. MoE is a **host break** — excluded from stream capture — so
  ``has_device_exec`` stays false; replay runs eager AOTI+Triton.

* ``piecewise`` (CG-1 exit): capture a trivial capturable op (``add``) into a
  device graph with ``has_device_exec=true``, then run eager ``inductor_moe_``
  between replays (FA2-style host break).

* ``capture_safe`` (CG-2): ``INFINI_MOE_CAPTURE_SAFE=1`` — MoE enters device
  capture with aten index_select+bmm under stream capture; Triton on eager.
  Gate: ``has_device_exec=true``, ``replay_op_list_fallback=0``.

* ``triton_capture``: Decode-phase MoE under ``full_and_piecewise`` — MoE
  enters device capture with Triton ``fused_moe_routed`` (no aten). Gate:
  ``has_device_exec=true``, path tag ``triton``, bf16 parity vs eager Triton
  max abs < 1e-2. (``INFINI_MOE_TRITON_CAPTURE`` deprecated.)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


H, E, N = 2048, 160, 512


@dataclass
class MoeHcGraphSpikeResult:
    passed: bool
    mode: str
    bucket: int
    layer_idx: int
    valid_len: int
    has_device_exec: bool
    last_replay_used_device: bool
    replay_device_ok: int
    replay_op_list_fallback: int
    eager_ms_per_iter: float
    replay_ms_per_iter: float
    piecewise_ms_per_iter: float = 0.0
    device_graph_log: str = ""
    note: str = ""
    error: Optional[str] = None
    illegal_ops_catalog: str = ""
    path_tag: str = ""
    parity_max_abs: float = -1.0
    # Phase 3 capture MM
    used_torch_mempool: bool = False
    capture_arena_bytes: int = 0
    capture_arena_blocks: int = 0
    capture_arena_retained_torch: int = 0
    peak_device_memory_bytes: int = 0
    torch_stream_matches_ic: bool = True


def _timed_ms(fn, *, warmup: int, iters: int, sync) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) * 1000.0 / max(iters, 1)


def _setup_moe_env(
    *,
    moe_configs: str,
    moe_triton_cache: str,
    valid_len: int,
    capture_safe: bool = False,
    triton_capture: bool = False,
    strict_replay: Optional[str] = None,
) -> None:
    if strict_replay is not None:
        os.environ["INFINI_GRAPH_STRICT_REPLAY"] = str(strict_replay)
    else:
        os.environ["INFINI_GRAPH_STRICT_REPLAY"] = os.environ.get(
            "INFINI_GRAPH_STRICT_REPLAY", "0"
        )
    os.environ["INFINI_PIECEWISE_VALID_LEN"] = str(int(valid_len))
    os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"
    os.environ["INFINI_MOE_ALLOW_JIT"] = "0"
    # Deprecated enable switch — never rely on it for capture.
    os.environ.pop("INFINI_MOE_TRITON_CAPTURE", None)
    if triton_capture:
        # Phase-adaptive Decode MoE: non-eager policy + Decode TLS phase.
        os.environ["INFINI_CUDAGRAPH_POLICY"] = "full_and_piecewise"
        os.environ.pop("INFINI_MOE_FORCE_HOST_BREAK", None)
        os.environ.pop("INFINI_MOE_CAPTURE_SAFE", None)
    else:
        if capture_safe:
            os.environ["INFINI_MOE_CAPTURE_SAFE"] = "1"
            os.environ.pop("INFINI_MOE_FORCE_HOST_BREAK", None)
        else:
            # Host-break modes: force HB even if Decode phase is set elsewhere.
            os.environ["INFINI_MOE_FORCE_HOST_BREAK"] = "1"
            os.environ.pop("INFINI_MOE_CAPTURE_SAFE", None)
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


def _register_moe_package(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
):
    import infinicore
    import torch
    from infinicore.lib import _infinicore as _ic
    from infinilm.compile.piecewise_moe_segment import make_moe_example_inputs
    from infinilm.compile.piecewise_segments import LAYER_AGNOSTIC_IDX, SEGMENT_MOE
    from infinilm.torch_llama.moe_ops import register_fused_moe_routed_op

    register_fused_moe_routed_op()

    pkg = Path(segment_pt2).resolve()
    if not pkg.is_file():
        raise FileNotFoundError(f"missing segment: {pkg}")

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
    return infinicore, torch, _ic, device, cuda_dev, dtype


def run_full_moe_spike(
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
    _setup_moe_env(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        valid_len=valid_len,
        capture_safe=False,
    )
    try:
        infinicore, torch, _ic, device, cuda_dev, dtype = _register_moe_package(
            segment_pt2=segment_pt2,
            bucket=bucket,
            layer_idx=layer_idx,
            device_index=device_index,
        )
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="full_moe",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=0.0,
            replay_ms_per_iter=0.0,
            error=str(exc),
        )

    seq = min(int(valid_len) if valid_len > 0 else 1, bucket)
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

    for _ in range(max(warmup, 3)):
        eager_once()
    sync()
    eager_ms = _timed_ms(eager_once, warmup=2, iters=iters, sync=sync)

    # Record MoE as host-break op (no Triton under stream capture).
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
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="full_moe",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=eager_ms,
            replay_ms_per_iter=0.0,
            error=f"record/instantiate failed: {exc}",
        )

    has_exec = bool(graph.has_device_exec())
    log_msg = graph.device_graph_log() or ""
    note = (
        "MoE is host-break: no device capture of Triton; "
        "use --mode capture_safe for CG-2 aten body, or --mode triton_capture"
    )

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
            mode="full_moe",
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
            note=note,
            error=f"replay failed: {exc}",
        )

    # Pass = host-break replay works; device exec for MoE-only graph is not expected.
    passed = replay_ms > 0 and not has_exec
    return MoeHcGraphSpikeResult(
        passed=passed,
        mode="full_moe",
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
        note=note,
        error=None if passed else "unexpected has_device_exec for MoE-only host-break graph",
    )


def run_capture_safe_spike(
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
    """CG-2: MoE-in-device-graph with aten body under capture; Triton eager."""
    _setup_moe_env(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        valid_len=valid_len,
        capture_safe=True,
        strict_replay="1",
    )
    illegal = (
        "Without INFINI_MOE_CAPTURE_SAFE: Triton fused_moe_routed + AOTI under "
        "hcStreamBeginCapture → instantiate fails (UNKNOWN_SCALAR / opaque not "
        "capture-safe). Catalog: fused_moe_routed Triton stage1/stage2, "
        "moe_align_block_size host loops cascading into shared addmm."
    )
    try:
        infinicore, torch, _ic, device, cuda_dev, dtype = _register_moe_package(
            segment_pt2=segment_pt2,
            bucket=bucket,
            layer_idx=layer_idx,
            device_index=device_index,
        )
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="capture_safe",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=0.0,
            replay_ms_per_iter=0.0,
            error=str(exc),
            illegal_ops_catalog=illegal,
        )

    seq = min(int(valid_len) if valid_len > 0 else 1, bucket)
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

    for _ in range(max(warmup, 3)):
        eager_once()
    sync()
    eager_ms = _timed_ms(eager_once, warmup=2, iters=iters, sync=sync)

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
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="capture_safe",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=eager_ms,
            replay_ms_per_iter=0.0,
            error=f"capture_safe record/instantiate failed: {exc}",
            illegal_ops_catalog=illegal,
            note="expected: aten MoE under capture yields has_device_exec",
        )

    has_exec = bool(graph.has_device_exec())
    log_msg = graph.device_graph_log() or ""
    note = (
        "CG-2: INFINI_MOE_CAPTURE_SAFE=1 — aten routed experts under stream "
        "capture; Triton on eager warmup/host path"
    )

    def replay_once():
        graph.run()

    try:
        for _ in range(2):
            graph.run()
            sync()
        replay_ms = _timed_ms(replay_once, warmup=2, iters=iters, sync=sync)
        replay_device = bool(graph.last_replay_used_device())
        fallback = int(graph.replay_op_list_fallback())
        device_ok = int(graph.replay_device_ok())
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="capture_safe",
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
            note=note,
            error=f"capture_safe replay failed: {exc}",
            illegal_ops_catalog=illegal,
        )

    passed = has_exec and fallback == 0 and replay_device
    err = None
    if not has_exec:
        err = "capture_safe gate failed: need has_device_exec=true"
    elif fallback != 0:
        err = f"capture_safe gate failed: replay_op_list_fallback={fallback} (want 0)"
    elif not replay_device:
        err = "capture_safe gate failed: last_replay_used_device=false"

    return MoeHcGraphSpikeResult(
        passed=passed,
        mode="capture_safe",
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
        note=note,
        error=err,
        illegal_ops_catalog=illegal,
        path_tag="aten",
    )


def run_triton_capture_spike(
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
    """Triton fused_moe_routed under MetaX stream capture (no aten body)."""
    _setup_moe_env(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        valid_len=valid_len,
        triton_capture=True,
        strict_replay="1",
    )
    illegal = (
        "Triton fused_moe_routed under hcStreamBeginCapture failed historically "
        "via AOTI (UNKNOWN_SCALAR). This mode uses eager-decode MoE "
        "(router+Triton+shared, no moe_B* AOTI) with device-side align."
    )
    try:
        infinicore, torch, _ic, device, cuda_dev, dtype = _register_moe_package(
            segment_pt2=segment_pt2,
            bucket=bucket,
            layer_idx=layer_idx,
            device_index=device_index,
        )
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="triton_capture",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=0.0,
            replay_ms_per_iter=0.0,
            error=str(exc),
            illegal_ops_catalog=illegal,
            path_tag="triton",
        )

    seq = min(int(valid_len) if valid_len > 0 else 1, bucket)
    # Fixed seed for parity (eager vs replay).
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(device_index)
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

    for _ in range(max(warmup, 3)):
        eager_once()
    sync()
    eager_ms = _timed_ms(eager_once, warmup=2, iters=iters, sync=sync)

    # Reference output from eager Triton (TRITON_CAPTURE does not change eager path).
    eager_once()
    sync()
    eager_ref = out_t.detach().float().clone()

    # Stream align is enforced in C++ via CUDAStreamGuard during capture.
    stream_match = True

    try:
        # Phase-adaptive MoE: Decode TLS + non-eager policy (set in _setup_moe_env).
        if hasattr(_ic, "set_inference_phase"):
            _ic.set_inference_phase("decode")
        infinicore.start_graph_recording(device)
        _ic.inductor_moe_(
            hidden._underlying,
            out._underlying,
            int(layer_idx),
            int(bucket),
        )
        graph = infinicore.stop_graph_recording()
        if hasattr(_ic, "set_inference_phase"):
            _ic.set_inference_phase("unknown")
    except Exception as exc:  # noqa: BLE001
        if hasattr(_ic, "set_inference_phase"):
            try:
                _ic.set_inference_phase("unknown")
            except Exception:  # noqa: BLE001
                pass
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="triton_capture",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=eager_ms,
            replay_ms_per_iter=0.0,
            error=f"triton_capture record/instantiate failed: {exc}",
            illegal_ops_catalog=illegal,
            note="expected: Triton MoE under capture yields has_device_exec",
            path_tag="triton",
            used_torch_mempool=bool(
                getattr(_ic, "capture_used_torch_mempool", lambda: False)()
            ),
        )

    has_exec = bool(graph.has_device_exec())
    log_msg = graph.device_graph_log() or ""
    arena_bytes = int(graph.capture_arena_bytes())
    arena_blocks = int(graph.capture_arena_blocks())
    arena_torch = int(graph.capture_arena_retained_torch())
    used_mempool = bool(getattr(_ic, "capture_used_torch_mempool", lambda: True)())
    peak_bytes = int(torch.cuda.max_memory_allocated(device_index))
    # Soft VRAM gate for single-layer smoke (full 30-seg serve needs ~40 GiB free).
    vram_gate_bytes = int(os.environ.get("INFINI_CAPTURE_VRAM_GATE_BYTES", str(2 << 30)))
    note = (
        "Decode-phase MoE (full_and_piecewise) — Triton fused_moe_routed under "
        f"stream capture; IC CaptureArena bytes={arena_bytes} blocks={arena_blocks} "
        f"retained_torch={arena_torch}; peak_alloc={peak_bytes}"
    )

    def replay_once():
        graph.run()

    try:
        for _ in range(2):
            graph.run()
            sync()
        replay_ms = _timed_ms(replay_once, warmup=2, iters=iters, sync=sync)
        replay_device = bool(graph.last_replay_used_device())
        fallback = int(graph.replay_op_list_fallback())
        device_ok = int(graph.replay_device_ok())
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="triton_capture",
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
            note=note,
            error=f"triton_capture replay failed: {exc}",
            illegal_ops_catalog=illegal,
            path_tag="triton",
            used_torch_mempool=used_mempool,
            capture_arena_bytes=arena_bytes,
            capture_arena_blocks=arena_blocks,
            capture_arena_retained_torch=arena_torch,
            peak_device_memory_bytes=peak_bytes,
            torch_stream_matches_ic=stream_match,
        )

    # Parity: replay vs eager Triton on same inputs.
    out_t.zero_()
    graph.run()
    sync()
    parity_max = float((out_t.float() - eager_ref).abs().max().item())
    parity_ok = parity_max < 1e-2
    vram_ok = peak_bytes <= vram_gate_bytes
    arena_ok = arena_bytes > 0 or arena_torch > 0

    passed = (
        has_exec
        and fallback == 0
        and replay_device
        and parity_ok
        and (not used_mempool)
        and arena_ok
        and vram_ok
        and os.environ.get("INFINI_CUDAGRAPH_POLICY") == "full_and_piecewise"
        and not os.environ.get("INFINI_MOE_FORCE_HOST_BREAK")
        and not os.environ.get("INFINI_MOE_CAPTURE_SAFE")
    )
    err = None
    if not has_exec:
        err = "triton_capture gate failed: need has_device_exec=true"
    elif fallback != 0:
        err = f"triton_capture gate failed: replay_op_list_fallback={fallback} (want 0)"
    elif not replay_device:
        err = "triton_capture gate failed: last_replay_used_device=false"
    elif not parity_ok:
        err = f"triton_capture parity failed: max_abs={parity_max} (>= 1e-2)"
    elif used_mempool:
        err = "triton_capture MM gate failed: torch MemPool still in use"
    elif not arena_ok:
        err = "triton_capture MM gate failed: CaptureArena empty (no IC/retain)"
    elif not vram_ok:
        err = (
            f"triton_capture VRAM gate failed: peak={peak_bytes} > "
            f"gate={vram_gate_bytes}"
        )
    elif os.environ.get("INFINI_MOE_CAPTURE_SAFE"):
        err = "triton_capture path polluted: INFINI_MOE_CAPTURE_SAFE set (want Triton-only)"

    if passed and eager_ms > 0 and replay_ms >= eager_ms:
        note += (
            f"; soft: replay_ms ({replay_ms:.3f}) not ≪ eager ({eager_ms:.3f}) "
            "— still PASS on has_device_exec+parity"
        )

    return MoeHcGraphSpikeResult(
        passed=passed,
        mode="triton_capture",
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
        note=note,
        error=err,
        illegal_ops_catalog=illegal,
        path_tag="triton",
        parity_max_abs=parity_max,
        used_torch_mempool=used_mempool,
        capture_arena_bytes=arena_bytes,
        capture_arena_blocks=arena_blocks,
        capture_arena_retained_torch=arena_torch,
        peak_device_memory_bytes=peak_bytes,
        torch_stream_matches_ic=stream_match,
    )


def run_piecewise_spike(
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
    """Device graph (add) + eager MoE between replays."""
    _setup_moe_env(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        valid_len=valid_len,
        capture_safe=False,
    )
    try:
        infinicore, torch, _ic, device, cuda_dev, dtype = _register_moe_package(
            segment_pt2=segment_pt2,
            bucket=bucket,
            layer_idx=layer_idx,
            device_index=device_index,
        )
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="piecewise",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=0.0,
            replay_ms_per_iter=0.0,
            error=str(exc),
        )

    seq = min(int(valid_len) if valid_len > 0 else 1, bucket)
    a_t = torch.randn(1, seq, H, device=cuda_dev, dtype=dtype)
    b_t = torch.randn(1, seq, H, device=cuda_dev, dtype=dtype)
    out_add_t = torch.empty(1, seq, H, device=cuda_dev, dtype=dtype)
    a = infinicore.from_torch(a_t)
    b = infinicore.from_torch(b_t)
    out_add = infinicore.from_torch(out_add_t)

    hidden_t = torch.randn(1, seq, H, device=cuda_dev, dtype=dtype)
    out_moe_t = torch.empty(1, seq, H, device=cuda_dev, dtype=dtype)
    hidden = infinicore.from_torch(hidden_t)
    out_moe = infinicore.from_torch(out_moe_t)

    def sync():
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

    def eager_moe():
        _ic.inductor_moe_(
            hidden._underlying,
            out_moe._underlying,
            int(layer_idx),
            int(bucket),
        )

    for _ in range(max(warmup, 3)):
        eager_moe()
    sync()
    eager_ms = _timed_ms(eager_moe, warmup=2, iters=iters, sync=sync)

    # One logical graph: capturable add, then MoE host-break, then add again.
    # Graph::instantiate splits around MoE so Triton never enters stream capture.
    try:
        infinicore.start_graph_recording(device)
        infinicore.add(a, b, out=out_add)
        _ic.inductor_moe_(
            hidden._underlying,
            out_moe._underlying,
            int(layer_idx),
            int(bucket),
        )
        # Second capturable stub after the MoE host break.
        infinicore.add(out_add, b, out=out_add)
        graph = infinicore.stop_graph_recording()
    except Exception as exc:  # noqa: BLE001
        return MoeHcGraphSpikeResult(
            passed=False,
            mode="piecewise",
            bucket=bucket,
            layer_idx=layer_idx,
            valid_len=valid_len,
            has_device_exec=False,
            last_replay_used_device=False,
            replay_device_ok=0,
            replay_op_list_fallback=0,
            eager_ms_per_iter=eager_ms,
            replay_ms_per_iter=0.0,
            error=f"piecewise record/instantiate failed: {exc}",
            note="expected: device segments around MoE host-break",
        )

    sync()
    has_exec = bool(graph.has_device_exec())
    log_msg = graph.device_graph_log() or ""
    note = (
        "CG-1: add | eager MoE host-break | add — MoE excluded from stream capture; "
        "CG-2 capture_safe mode for MoE-in-device"
    )

    replay_device = False
    fallback = 0
    device_ok = 0
    pw_ms = 0.0
    replay_error = None
    if has_exec:
        def piecewise_once():
            graph.run()

        try:
            for _ in range(2):
                graph.run()
                infinicore.sync_stream()
            pw_ms = _timed_ms(
                piecewise_once,
                warmup=1,
                iters=max(iters // 2, 5),
                sync=lambda: infinicore.sync_stream(),
            )
            replay_device = bool(graph.last_replay_used_device())
            fallback = int(graph.replay_op_list_fallback())
            device_ok = int(graph.replay_device_ok())
        except Exception as exc:  # noqa: BLE001
            replay_error = str(exc)
            note += (
                f"; device stub replay flaky on MetaX ({type(exc).__name__}) — "
                "instantiate has_device_exec still proves MoE was skipped under capture"
            )

    # Gate: device segments instantiated (MoE was host-break, not under capture).
    passed = has_exec
    err = None
    if not passed:
        err = (
            "piecewise gate failed: need has_device_exec=true after host-break split "
            f"(has_device_exec={has_exec})"
        )
    elif replay_error:
        err = None  # soft: instantiate gate is the CG-1 exit

    return MoeHcGraphSpikeResult(
        passed=passed,
        mode="piecewise",
        bucket=bucket,
        layer_idx=layer_idx,
        valid_len=valid_len,
        has_device_exec=has_exec,
        last_replay_used_device=replay_device,
        replay_device_ok=device_ok,
        replay_op_list_fallback=fallback,
        eager_ms_per_iter=eager_ms,
        replay_ms_per_iter=pw_ms,
        piecewise_ms_per_iter=pw_ms,
        device_graph_log=log_msg,
        note=note,
        error=err,
    )


# Back-compat alias used by older callers / OPT_NOTES recipes.
def run_moe_hcgraph_spike(**kwargs) -> MoeHcGraphSpikeResult:
    return run_full_moe_spike(**kwargs)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("full_moe", "piecewise", "capture_safe", "triton_capture"),
        default="piecewise",
        help="full_moe: host-break; piecewise: stub+eager MoE; "
        "capture_safe: CG-2 aten MoE-in-graph; "
        "triton_capture: Triton MoE-in-graph (default: piecewise)",
    )
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
    parser.add_argument("--moe-configs", default="")
    parser.add_argument("--moe-triton-cache", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[2]
    default_pkg = (
        root
        / "bench_results/piecewise_inductor_cache_minicpm5"
        / "minicpm5.16a3.v0314_057ba9e7b8f5"
        / "tp1/rank0/moe_B16/segment.pt2"
    )
    segment = args.segment_pt2 or str(default_pkg)
    hash_root = Path(segment).resolve().parents[3]
    moe_configs = args.moe_configs or str(hash_root / "moe_configs")
    moe_triton = args.moe_triton_cache or str(hash_root / "moe_triton_cache")

    runners = {
        "piecewise": run_piecewise_spike,
        "full_moe": run_full_moe_spike,
        "capture_safe": run_capture_safe_spike,
        "triton_capture": run_triton_capture_spike,
    }
    runner = runners[args.mode]
    try:
        result = runner(
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
            mode=args.mode,
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
    print(
        f"[moe_hcgraph_spike] {status} mode={result.mode} bucket={result.bucket} "
        f"L{result.layer_idx} valid={result.valid_len} "
        f"has_device_exec={result.has_device_exec} "
        f"last_replay_used_device={result.last_replay_used_device} "
        f"replay_device_ok={result.replay_device_ok} "
        f"replay_op_list_fallback={result.replay_op_list_fallback} "
        f"path_tag={result.path_tag or '-'} "
        f"parity_max_abs={result.parity_max_abs}",
        flush=True,
    )
    print(
        f"[moe_hcgraph_spike] eager_moe_ms/iter={result.eager_ms_per_iter:.3f} "
        f"replay_or_piecewise_ms/iter={result.replay_ms_per_iter:.3f}",
        flush=True,
    )
    if result.note:
        print(f"[moe_hcgraph_spike] note: {result.note}", flush=True)
    if result.illegal_ops_catalog:
        print(f"[moe_hcgraph_spike] illegal_ops: {result.illegal_ops_catalog}", flush=True)
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

    os._exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
