#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""P8a — FA co-capture diagnose smoke (GPU0/2; never GPU3 fair-grid).

Cells
-----
* ``moe_span`` (green expected): Decode-phase MoE under ``full_and_piecewise`` —
  record one Triton MoE into a device graph (known good after Phase 5).
  (``INFINI_MOE_TRITON_CAPTURE`` deprecated.)

* ``fa_hostbreak_sandwich`` (green expected): MoE | FA(host-break) | MoE —
  FA stays outside ``hcStreamBeginCapture``; two device segments + FA HostOp.
  Mirrors production span-fuse handoff shape.

* ``fa_alone`` (red expected): ``INFINI_FA_FORCE_CAPTURE=1`` — FA alone under
  stream capture. Documents exact EndCapture / HTC / IllegalAddress fingerprint.

* ``fa_force_sandwich`` (red expected): MoE | FA(force-capture) | MoE — FA
  between two Triton spans under one recording; co-capture diagnose.

Artifacts: write ``--json-out``; campaign aggregates under ``p8a_fa_smoke/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


H, E, N = 2048, 160, 512
# MiniCPM5 decode dims (config: 32/2/128; serve paged block=256).
HQ, HKV, HD, BLOCK = 32, 2, 128, 256


@dataclass
class FaCaptureCellResult:
    cell: str
    passed: bool
    expect: str  # "green" | "red"
    color: str  # "green" | "red" | "amber"
    has_device_exec: bool = False
    device_segment_count: int = 0
    device_graph_log: str = ""
    fault_phase: str = ""  # recording | instantiate | replay | none
    fault: Optional[str] = None
    note: str = ""
    force_fa_capture: bool = False
    triton_capture: bool = False


def _setup_common(
    *,
    moe_configs: str,
    moe_triton_cache: str,
    triton_capture: bool,
    force_fa: bool,
) -> None:
    os.environ["INFINI_GRAPH_STRICT_REPLAY"] = "1"
    os.environ["INFINI_PIECEWISE_VALID_LEN"] = "1"
    os.environ["INFINI_PIECEWISE_INDUCTOR_SEGMENT"] = "1"
    os.environ["INFINI_MOE_ALLOW_JIT"] = "0"
    os.environ.pop("INFINI_MOE_TRITON_CAPTURE", None)
    if triton_capture:
        os.environ["INFINI_CUDAGRAPH_POLICY"] = "full_and_piecewise"
        os.environ.pop("INFINI_MOE_FORCE_HOST_BREAK", None)
        os.environ.pop("INFINI_MOE_CAPTURE_SAFE", None)
    else:
        os.environ["INFINI_MOE_FORCE_HOST_BREAK"] = "1"
        os.environ.pop("INFINI_MOE_CAPTURE_SAFE", None)
    if force_fa:
        os.environ["INFINI_FA_FORCE_CAPTURE"] = "1"
    else:
        os.environ.pop("INFINI_FA_FORCE_CAPTURE", None)
    if moe_configs:
        os.environ["INFINI_MOE_CONFIGS"] = moe_configs
    if moe_triton_cache:
        os.environ["INFINI_MOE_TRITON_CACHE"] = moe_triton_cache
        os.environ["TRITON_CACHE_DIR"] = moe_triton_cache


def _register_moe(
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


def _make_fa_tensors(
    infinicore,
    torch,
    cuda_dev,
    dtype,
    *,
    kv_len: int = 512,
    hq: int = HQ,
    hkv: int = HKV,
    hd: int = HD,
    block: int = BLOCK,
):
    num_blocks = max(1, (kv_len + block - 1) // block)
    q_t = torch.randn(1, 1, hq, hd, device=cuda_dev, dtype=dtype)
    k_t = torch.randn(num_blocks, block, hkv, hd, device=cuda_dev, dtype=dtype)
    v_t = torch.randn(num_blocks, block, hkv, hd, device=cuda_dev, dtype=dtype)
    seqlens_t = torch.tensor([kv_len], device=cuda_dev, dtype=torch.int32)
    bt_t = torch.arange(num_blocks, device=cuda_dev, dtype=torch.int32).view(1, -1)
    q = infinicore.from_torch(q_t.contiguous())
    k = infinicore.from_torch(k_t.contiguous())
    v = infinicore.from_torch(v_t.contiguous())
    seqlens = infinicore.from_torch(seqlens_t.contiguous())
    bt = infinicore.from_torch(bt_t.contiguous())
    scale = 1.0 / math.sqrt(float(hd))
    return q, k, v, seqlens, bt, scale


def _make_moe_tensors(infinicore, torch, _ic, cuda_dev, dtype, *, bucket: int, layer_idx: int):
    hidden_t = torch.randn(1, 1, H, device=cuda_dev, dtype=dtype)
    out_t = torch.empty(1, 1, H, device=cuda_dev, dtype=dtype)
    hidden = infinicore.from_torch(hidden_t)
    out = infinicore.from_torch(out_t)

    def run_moe():
        _ic.inductor_moe_(
            hidden._underlying,
            out._underlying,
            int(layer_idx),
            int(bucket),
        )

    return hidden, out, run_moe


def _classify_fault(exc: BaseException) -> tuple[str, str]:
    msg = f"{type(exc).__name__}: {exc}"
    low = msg.lower()
    if "must not run under hcstream capture" in low:
        return "recording", msg
    if "endcapture" in low or "begincapture" in low or "illegaladdress" in low:
        return "instantiate", msg
    if "htc" in low or "memory violation" in low or "atu" in low:
        return "instantiate", msg
    if "instantiate" in low or "capture" in low:
        return "instantiate", msg
    return "unknown", msg


def _finish(
    *,
    cell: str,
    expect: str,
    graph,
    fault_phase: str = "none",
    fault: Optional[str] = None,
    note: str = "",
    force_fa: bool = False,
    triton: bool = False,
) -> FaCaptureCellResult:
    has_exec = bool(graph.has_device_exec()) if graph is not None else False
    nseg = int(graph.device_segment_count()) if graph is not None else 0
    log = (graph.device_graph_log() or "") if graph is not None else ""
    # Green cells must succeed; red cells "pass" the diagnose gate when they
    # fail with a capture fault (or soft-refuse) — i.e. evidence recorded.
    if expect == "green":
        ok = fault is None and has_exec
        color = "green" if ok else "red"
        passed = ok
    else:
        # Red expected: any capture-time failure or soft refuse counts as
        # documented blocker (diagnose PASS). Unexpected success → amber.
        if fault is not None:
            color = "red"
            passed = True  # diagnose evidence collected
        elif has_exec:
            color = "amber"
            passed = False
            note = (note + "; " if note else "") + "unexpected FA-in-capture success"
        else:
            color = "red"
            passed = True
            fault = fault or "no device exec (FA excluded or capture failed softly)"
    return FaCaptureCellResult(
        cell=cell,
        passed=passed,
        expect=expect,
        color=color,
        has_device_exec=has_exec,
        device_segment_count=nseg,
        device_graph_log=log[:800],
        fault_phase=fault_phase,
        fault=fault,
        note=note,
        force_fa_capture=force_fa,
        triton_capture=triton,
    )


def _set_decode_phase(_ic) -> None:
    """Enable phase-adaptive MoE in-graph for smoke cells that need Triton capture."""
    if hasattr(_ic, "set_inference_phase"):
        _ic.set_inference_phase("decode")


def _clear_phase(_ic) -> None:
    if hasattr(_ic, "set_inference_phase"):
        _ic.set_inference_phase("unknown")


def run_moe_span(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> FaCaptureCellResult:
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=True,
        force_fa=False,
    )
    _ic = None
    try:
        infinicore, torch, _ic, device, cuda_dev, dtype = _register_moe(
            segment_pt2=segment_pt2,
            bucket=bucket,
            layer_idx=layer_idx,
            device_index=device_index,
        )
        _, _, run_moe = _make_moe_tensors(
            infinicore, torch, _ic, cuda_dev, dtype, bucket=bucket, layer_idx=layer_idx
        )
        for _ in range(3):
            run_moe()
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()
        _set_decode_phase(_ic)
        infinicore.start_graph_recording(device)
        run_moe()
        graph = infinicore.stop_graph_recording()
        _clear_phase(_ic)
        graph.run()
        infinicore.sync_stream()
        return _finish(
            cell="moe_span",
            expect="green",
            graph=graph,
            note="Decode-phase Triton MoE under hcStreamBeginCapture",
            triton=True,
        )
    except Exception as exc:  # noqa: BLE001
        if _ic is not None:
            _clear_phase(_ic)
        phase, msg = _classify_fault(exc)
        return FaCaptureCellResult(
            cell="moe_span",
            passed=False,
            expect="green",
            color="red",
            fault_phase=phase,
            fault=msg,
            note="expected green MoE-in-span",
            triton_capture=True,
        )


def run_fa_hostbreak_sandwich(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> FaCaptureCellResult:
    """MoE | FA(host-break) | MoE — production handoff shape; FA never in capture."""
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=True,
        force_fa=False,
    )
    _ic = None
    try:
        infinicore, torch, _ic, device, cuda_dev, dtype = _register_moe(
            segment_pt2=segment_pt2,
            bucket=bucket,
            layer_idx=layer_idx,
            device_index=device_index,
        )
        _, _, run_moe = _make_moe_tensors(
            infinicore, torch, _ic, cuda_dev, dtype, bucket=bucket, layer_idx=layer_idx
        )
        q, k, v, seqlens, bt, scale = _make_fa_tensors(infinicore, torch, cuda_dev, dtype)
        for _ in range(2):
            run_moe()
            infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

        _set_decode_phase(_ic)
        infinicore.start_graph_recording(device)
        run_moe()
        infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)  # host-break
        run_moe()
        graph = infinicore.stop_graph_recording()
        _clear_phase(_ic)
        graph.run()
        infinicore.sync_stream()
        nseg = int(graph.device_segment_count())
        ok_segs = nseg >= 2
        result = _finish(
            cell="fa_hostbreak_sandwich",
            expect="green",
            graph=graph,
            note=f"MoE|FA_hostbreak|MoE handoff; device_segments={nseg}",
            triton=True,
        )
        if result.passed and not ok_segs:
            result.passed = False
            result.color = "red"
            result.fault = f"expected ≥2 device segments around FA host-break, got {nseg}"
        return result
    except Exception as exc:  # noqa: BLE001
        if _ic is not None:
            _clear_phase(_ic)
        phase, msg = _classify_fault(exc)
        return FaCaptureCellResult(
            cell="fa_hostbreak_sandwich",
            passed=False,
            expect="green",
            color="red",
            fault_phase=phase,
            fault=msg,
            note="FA host-break sandwich should instantiate",
            triton_capture=True,
        )


def run_fa_alone(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> FaCaptureCellResult:
    """FA alone under capture with INFINI_FA_FORCE_CAPTURE=1.

    Uses MiniCPM5 shapes; records **28 FA ops** (layer count) in one logical
    graph to stress EndCapture — historically HTC / IllegalAddress.
    """
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=False,
        force_fa=True,
    )
    n_layers = int(os.environ.get("INFINI_FA_SMOKE_LAYERS", "28"))
    try:
        import infinicore
        import torch

        device = infinicore.device("cuda", int(device_index))
        infinicore.set_device(device)
        cuda_dev = f"cuda:{int(device_index)}"
        dtype = torch.bfloat16
        q, k, v, seqlens, bt, scale = _make_fa_tensors(infinicore, torch, cuda_dev, dtype)
        for _ in range(2):
            infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

        infinicore.start_graph_recording(device)
        for _ in range(n_layers):
            infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        graph = infinicore.stop_graph_recording()
        try:
            graph.run()
            infinicore.sync_stream()
        except Exception as replay_exc:  # noqa: BLE001
            phase, msg = _classify_fault(replay_exc)
            return FaCaptureCellResult(
                cell="fa_alone",
                passed=True,
                expect="red",
                color="red",
                has_device_exec=bool(graph.has_device_exec()),
                device_segment_count=int(graph.device_segment_count()),
                device_graph_log=(graph.device_graph_log() or "")[:800],
                fault_phase="replay",
                fault=msg,
                note=f"FA×{n_layers} MiniCPM5-shaped under FORCE_CAPTURE — fault at replay",
                force_fa_capture=True,
            )
        return _finish(
            cell="fa_alone",
            expect="red",
            graph=graph,
            note=f"FA×{n_layers} MiniCPM5-shaped under FORCE_CAPTURE",
            force_fa=True,
        )
    except Exception as exc:  # noqa: BLE001
        phase, msg = _classify_fault(exc)
        return FaCaptureCellResult(
            cell="fa_alone",
            passed=True,  # red evidence
            expect="red",
            color="red",
            fault_phase=phase,
            fault=msg + "\n" + traceback.format_exc()[-400:],
            note=f"FA×{n_layers} MiniCPM5-shaped in-recording fault (expected)",
            force_fa_capture=True,
        )


def run_fa_soft_refuse(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> FaCaptureCellResult:
    """Production guard: FA under capture without FORCE → soft refuse string.

    Forces host_break off only for the *recording split* by setting
    INFINI_FA_FORCE_CAPTURE during record, then clears it before replay so the
    run() path hits the soft refuse. Simpler: call FA while capturing with
    FORCE on for host_break=false, but invoke run via a nested path…

    Practical approach: with FORCE unset, FA is host_break so it never enters
    BeginCapture. Soft refuse is exercised by FORCE=1 for host_break_=false
    then temporarily unsetting FORCE mid-instantiate is not possible.

    Instead: document production refuse by calling FA while
    ``is_device_stream_capturing`` is simulated — we trigger the refuse by
    recording with FORCE (host_break false) then in a *second* graph with FORCE
    off we cannot get FA into capture.

    So this cell explicitly records with FORCE=1 (so FA enters the device
    segment), then **replays after unsetting FORCE** so Graph::run hits the
    soft refuse throw inside FA::run under capturing? No — replay is not
    capturing.

    Soft refuse only fires during instantiate capture. So: FORCE must be on
    for host_break=false, and refuse is skipped when FORCE is on.

    Conclusion: soft refuse is only hit if host_break=false AND FORCE=0 —
    impossible with current gating (host_break follows FORCE).

    Cell instead verifies: FORCE=0 → FA alone yields **no** device capture
    (host_break HostOp only) — production-safe exclusion fingerprint.
    """
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=False,
        force_fa=False,
    )
    try:
        import infinicore
        import torch

        device = infinicore.device("cuda", int(device_index))
        infinicore.set_device(device)
        cuda_dev = f"cuda:{int(device_index)}"
        dtype = torch.bfloat16
        q, k, v, seqlens, bt, scale = _make_fa_tensors(infinicore, torch, cuda_dev, dtype)
        for _ in range(2):
            infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

        infinicore.start_graph_recording(device)
        infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        graph = infinicore.stop_graph_recording()
        has_exec = bool(graph.has_device_exec())
        # Production exclusion: FA is host_break → no device segment.
        if not has_exec:
            return FaCaptureCellResult(
                cell="fa_soft_refuse",
                passed=True,
                expect="red",
                color="red",
                has_device_exec=False,
                device_segment_count=int(graph.device_segment_count()),
                fault_phase="recording",
                fault=(
                    "FA host_break exclusion (FORCE unset): no hcStreamBeginCapture of "
                    "mha_fwd_kvcache; Graph treats FA as HostOp. Runtime soft-refuse "
                    "string if FA ever runs under capture: "
                    "'MhaKVCache::run: FA2 mha_fwd_kvcache must not run under hcStream capture'"
                ),
                note="production FA exclusion (not in-recording)",
                force_fa_capture=False,
            )
        return FaCaptureCellResult(
            cell="fa_soft_refuse",
            passed=False,
            expect="red",
            color="amber",
            has_device_exec=True,
            device_segment_count=int(graph.device_segment_count()),
            fault_phase="none",
            fault=None,
            note="unexpected device capture of FA with FORCE unset",
            force_fa_capture=False,
        )
    except Exception as exc:  # noqa: BLE001
        phase, msg = _classify_fault(exc)
        return FaCaptureCellResult(
            cell="fa_soft_refuse",
            passed=True,
            expect="red",
            color="red",
            fault_phase=phase,
            fault=msg,
            note="FA exclusion / refuse fingerprint",
            force_fa_capture=False,
        )


def run_fa_force_sandwich(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> FaCaptureCellResult:
    """MoE | FA(force) | MoE — co-capture; red expected."""
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=True,
        force_fa=True,
    )
    _ic = None
    try:
        infinicore, torch, _ic, device, cuda_dev, dtype = _register_moe(
            segment_pt2=segment_pt2,
            bucket=bucket,
            layer_idx=layer_idx,
            device_index=device_index,
        )
        _, _, run_moe = _make_moe_tensors(
            infinicore, torch, _ic, cuda_dev, dtype, bucket=bucket, layer_idx=layer_idx
        )
        q, k, v, seqlens, bt, scale = _make_fa_tensors(infinicore, torch, cuda_dev, dtype)
        for _ in range(2):
            run_moe()
            infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

        _set_decode_phase(_ic)
        infinicore.start_graph_recording(device)
        run_moe()
        infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        run_moe()
        graph = infinicore.stop_graph_recording()
        _clear_phase(_ic)
        try:
            graph.run()
            infinicore.sync_stream()
        except Exception as replay_exc:  # noqa: BLE001
            phase, msg = _classify_fault(replay_exc)
            return FaCaptureCellResult(
                cell="fa_force_sandwich",
                passed=True,
                expect="red",
                color="red",
                has_device_exec=bool(graph.has_device_exec()),
                device_segment_count=int(graph.device_segment_count()),
                device_graph_log=(graph.device_graph_log() or "")[:800],
                fault_phase="replay",
                fault=msg,
                note="FA between MoE spans under FORCE_CAPTURE — fault at replay",
                force_fa_capture=True,
                triton_capture=True,
            )
        return _finish(
            cell="fa_force_sandwich",
            expect="red",
            graph=graph,
            note="FA co-capture with MoE under FORCE_CAPTURE",
            force_fa=True,
            triton=True,
        )
    except Exception as exc:  # noqa: BLE001
        if _ic is not None:
            _clear_phase(_ic)
        phase, msg = _classify_fault(exc)
        return FaCaptureCellResult(
            cell="fa_force_sandwich",
            passed=True,
            expect="red",
            color="red",
            fault_phase=phase,
            fault=msg + "\n" + traceback.format_exc()[-400:],
            note="FA co-capture fault (expected)",
            force_fa_capture=True,
            triton_capture=True,
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        choices=(
            "moe_span",
            "fa_hostbreak_sandwich",
            "fa_soft_refuse",
            "fa_alone",
            "fa_force_sandwich",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--segment-pt2", default="")
    parser.add_argument("--bucket", type=int, default=16)
    parser.add_argument("--layer-idx", type=int, default=0)
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
        "moe_span": run_moe_span,
        "fa_hostbreak_sandwich": run_fa_hostbreak_sandwich,
        "fa_soft_refuse": run_fa_soft_refuse,
        "fa_alone": run_fa_alone,
        "fa_force_sandwich": run_fa_force_sandwich,
    }
    cells = list(runners.keys()) if args.cell == "all" else [args.cell]
    results: list[dict[str, Any]] = []
    all_ok = True
    for name in cells:
        # Fresh process-local env per cell; FA force must not leak into green cells.
        print(f"[fa_capture_smoke] === cell={name} ===", flush=True)
        try:
            r = runners[name](
                segment_pt2=segment,
                bucket=args.bucket,
                layer_idx=args.layer_idx,
                device_index=args.device_index,
                moe_configs=moe_configs,
                moe_triton_cache=moe_triton,
            )
        except Exception as exc:  # noqa: BLE001
            r = FaCaptureCellResult(
                cell=name,
                passed=False,
                expect="green" if "fa_alone" not in name and "force" not in name else "red",
                color="red",
                fault_phase="unknown",
                fault=str(exc),
            )
        d = asdict(r)
        results.append(d)
        status = "PASS" if r.passed else "FAIL"
        print(
            f"[fa_capture_smoke] {status} cell={r.cell} color={r.color} expect={r.expect} "
            f"has_device_exec={r.has_device_exec} segs={r.device_segment_count} "
            f"fault_phase={r.fault_phase or '-'}",
            flush=True,
        )
        if r.fault:
            print(f"[fa_capture_smoke] fault: {r.fault[:500]}", flush=True)
        if r.note:
            print(f"[fa_capture_smoke] note: {r.note}", flush=True)
        if not r.passed:
            all_ok = False

    payload = {
        "cells": results,
        "diagnose_gate": all_ok,
        "device_index": args.device_index,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[fa_capture_smoke] wrote {out}", flush=True)

    os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
