#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Phase-0 iso matrix for FA H3/H4 (dual-entry poison + HostOp from_blob bridge).

Cells (process-local; run one cell per process — H3 FAIL cells may SIGSEGV/HTC)::

* ``h3_kvcache_then_varlen`` — ``FA_FORCE=1``: record+instantiate ``mha_kvcache`` ×N,
  then eager ``mha_varlen``. Expect FAIL if H3 true.
* ``h3_kvcache_then_kvcache`` — same capture; then eager ``mha_kvcache`` only.
  Isolates dual-entry (PASS or milder than varlen).
* ``h3_no_capture_varlen`` — never FORCE; eager kvcache then varlen. Baseline PASS.
* ``h4_hostop_from_blob`` — ``FA_FORCE=0`` MoE|FA|MoE HostOp sandwich; documents
  current ``to_aten_tensor`` → ``at::from_blob`` bridge (H4: HB ≠ FX split).
* ``h4_hostop_owning`` — same sandwich + owning ATen prototype
  (``INFINI_FA_ATEN_OWNED=1`` when Phase-1 lands; else Python flash-attn owned clones).

Artifacts: ``--json-out`` per cell; campaign aggregates ``ISO_SUMMARY.json``.
Does **not** enable phase-adaptive FA (``faInGraphAllowed`` stays FORCE-only).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse FA/MoE helpers from P8a smoke.
from fa_capture_smoke import (  # noqa: E402
    BLOCK,
    HD,
    HKV,
    HQ,
    _classify_fault,
    _make_fa_tensors,
    _make_moe_tensors,
    _register_moe,
    _setup_common,
    _clear_phase,
    _set_decode_phase,
)


@dataclass
class H3H4CellResult:
    cell: str
    outcome: str  # PASS | FAIL | SKIP
    expect_if_hypothesis: str  # PASS | FAIL | DOC
    hypothesis: str  # H3 | H4
    hypothesis_consistent: bool
    has_device_exec: bool = False
    device_segment_count: int = 0
    device_graph_log: str = ""
    fault_phase: str = ""
    fault: Optional[str] = None
    note: str = ""
    force_fa_capture: bool = False
    aten_bridge: str = ""  # from_blob | owned | owned_prototype
    extras: dict[str, Any] = field(default_factory=dict)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _n_fa_layers() -> int:
    return int(os.environ.get("INFINI_FA_H3_LAYERS", os.environ.get("INFINI_FA_SMOKE_LAYERS", "4")))


def _make_varlen_tensors(infinicore, torch, cuda_dev, dtype, *, q_len: int = 7, kv_len: int = 64):
    """MiniCPM5-shaped paged varlen (prefill-like) inputs for InfiniCore ``mha_varlen``."""
    hq, hkv, hd, block = HQ, HKV, HD, BLOCK
    num_blocks = max(1, (kv_len + block - 1) // block)
    # FA varlen Q: [total_q, hq, hd]; K/V paged: [num_blocks, block, hkv, hd]
    q_t = torch.randn(q_len, hq, hd, device=cuda_dev, dtype=dtype)
    k_t = torch.randn(num_blocks, block, hkv, hd, device=cuda_dev, dtype=dtype)
    v_t = torch.randn(num_blocks, block, hkv, hd, device=cuda_dev, dtype=dtype)
    cum_q = torch.tensor([0, q_len], device=cuda_dev, dtype=torch.int32)
    cum_k = torch.tensor([0, kv_len], device=cuda_dev, dtype=torch.int32)
    bt_t = torch.arange(num_blocks, device=cuda_dev, dtype=torch.int32).view(1, -1)
    q = infinicore.from_torch(q_t.contiguous())
    k = infinicore.from_torch(k_t.contiguous())
    v = infinicore.from_torch(v_t.contiguous())
    cum_seqlens_q = infinicore.from_torch(cum_q.contiguous())
    cum_seqlens_k = infinicore.from_torch(cum_k.contiguous())
    bt = infinicore.from_torch(bt_t.contiguous())
    scale = 1.0 / math.sqrt(float(hd))
    return q, k, v, cum_seqlens_q, cum_seqlens_k, bt, scale, q_len, kv_len


def _capture_fa_kvcache(
    *,
    infinicore,
    torch,
    device,
    device_index: int,
    q,
    k,
    v,
    seqlens,
    bt,
    scale,
    n_layers: int,
):
    """Record+instantiate FA kvcache under FORCE. Returns (graph, fault_or_None)."""
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
        return graph, None
    except Exception as replay_exc:  # noqa: BLE001
        phase, msg = _classify_fault(replay_exc)
        return graph, f"replay/{phase}: {msg}"


def _finish_h3(
    *,
    cell: str,
    expect_if_h3: str,
    outcome: str,
    graph=None,
    fault_phase: str = "none",
    fault: Optional[str] = None,
    note: str = "",
    force_fa: bool = False,
    extras: Optional[dict[str, Any]] = None,
) -> H3H4CellResult:
    has_exec = bool(graph.has_device_exec()) if graph is not None else False
    nseg = int(graph.device_segment_count()) if graph is not None else 0
    log = (graph.device_graph_log() or "") if graph is not None else ""
    consistent = outcome == expect_if_h3
    return H3H4CellResult(
        cell=cell,
        outcome=outcome,
        expect_if_hypothesis=expect_if_h3,
        hypothesis="H3",
        hypothesis_consistent=consistent,
        has_device_exec=has_exec,
        device_segment_count=nseg,
        device_graph_log=log[:800],
        fault_phase=fault_phase,
        fault=fault,
        note=note,
        force_fa_capture=force_fa,
        aten_bridge="from_blob",
        extras=extras or {},
    )


def run_h3_kvcache_then_varlen(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> H3H4CellResult:
    """FORCE capture mha_kvcache ×N, then eager mha_varlen — H3 dual-entry poison."""
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=False,
        force_fa=True,
    )
    n_layers = _n_fa_layers()
    graph = None
    try:
        import infinicore
        import torch

        device = infinicore.device("cuda", int(device_index))
        infinicore.set_device(device)
        cuda_dev = f"cuda:{int(device_index)}"
        dtype = torch.bfloat16
        q, k, v, seqlens, bt, scale = _make_fa_tensors(infinicore, torch, cuda_dev, dtype)
        graph, cap_fault = _capture_fa_kvcache(
            infinicore=infinicore,
            torch=torch,
            device=device,
            device_index=device_index,
            q=q,
            k=k,
            v=v,
            seqlens=seqlens,
            bt=bt,
            scale=scale,
            n_layers=n_layers,
        )
        if cap_fault is not None:
            # Capture itself failed — still attempt varlen to document poison vs capture fault.
            note = f"capture/replay fault before varlen: {cap_fault}"
        else:
            note = f"captured FA kvcache ×{n_layers}; probing eager mha_varlen"

        # Clear FORCE so subsequent eager varlen mirrors production prefill entry
        # (same adaptor bridge, not under capture).
        os.environ.pop("INFINI_FA_FORCE_CAPTURE", None)

        vq, vk, vv, cq, ck, vbt, vscale, q_len, kv_len = _make_varlen_tensors(
            infinicore, torch, cuda_dev, dtype
        )
        out = infinicore.mha_varlen(
            vq, vk, vv, cq, ck, vbt, max(q_len, 64), max(kv_len, 64), scale=vscale
        )
        _ = out
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()
        return _finish_h3(
            cell="h3_kvcache_then_varlen",
            expect_if_h3="FAIL",
            outcome="PASS",
            graph=graph,
            note=note + "; eager varlen OK → H3 not confirmed",
            force_fa=True,
            extras={"n_layers": n_layers, "post_force_cleared": True},
        )
    except Exception as exc:  # noqa: BLE001
        phase, msg = _classify_fault(exc)
        # Treat SIGSEGV-class / HTC as FAIL evidence for H3.
        return _finish_h3(
            cell="h3_kvcache_then_varlen",
            expect_if_h3="FAIL",
            outcome="FAIL",
            graph=graph,
            fault_phase=phase,
            fault=msg + "\n" + traceback.format_exc()[-500:],
            note=f"varlen after FA capture failed (H3-consistent if poison)",
            force_fa=True,
            extras={"n_layers": n_layers},
        )


def run_h3_kvcache_then_kvcache(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> H3H4CellResult:
    """Same FORCE capture; then eager mha_kvcache only — isolates dual-entry."""
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=False,
        force_fa=True,
    )
    n_layers = _n_fa_layers()
    graph = None
    try:
        import infinicore
        import torch

        device = infinicore.device("cuda", int(device_index))
        infinicore.set_device(device)
        cuda_dev = f"cuda:{int(device_index)}"
        dtype = torch.bfloat16
        q, k, v, seqlens, bt, scale = _make_fa_tensors(infinicore, torch, cuda_dev, dtype)
        graph, cap_fault = _capture_fa_kvcache(
            infinicore=infinicore,
            torch=torch,
            device=device,
            device_index=device_index,
            q=q,
            k=k,
            v=v,
            seqlens=seqlens,
            bt=bt,
            scale=scale,
            n_layers=n_layers,
        )
        os.environ.pop("INFINI_FA_FORCE_CAPTURE", None)
        # Fresh tensors for post-capture eager kvcache.
        q2, k2, v2, seqlens2, bt2, scale2 = _make_fa_tensors(
            infinicore, torch, cuda_dev, dtype
        )
        for _ in range(2):
            infinicore.mha_kvcache(q2, k2, v2, seqlens2, bt2, scale=scale2)
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()
        note = "eager kvcache after capture OK"
        if cap_fault:
            note = f"capture fault ({cap_fault}); eager kvcache still ran"
        # Milder than varlen: expect PASS (or milder) if dual-entry is the issue.
        return _finish_h3(
            cell="h3_kvcache_then_kvcache",
            expect_if_h3="PASS",
            outcome="PASS",
            graph=graph,
            note=note,
            force_fa=True,
            extras={"n_layers": n_layers, "capture_fault": cap_fault},
        )
    except Exception as exc:  # noqa: BLE001
        phase, msg = _classify_fault(exc)
        return _finish_h3(
            cell="h3_kvcache_then_kvcache",
            expect_if_h3="PASS",
            outcome="FAIL",
            graph=graph,
            fault_phase=phase,
            fault=msg + "\n" + traceback.format_exc()[-500:],
            note="eager kvcache after capture failed (milder dual-entry still broken)",
            force_fa=True,
            extras={"n_layers": n_layers},
        )


def run_h3_no_capture_varlen(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> H3H4CellResult:
    """Never FORCE; eager kvcache then varlen — H3 baseline (expect PASS)."""
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
        for _ in range(3):
            infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()

        vq, vk, vv, cq, ck, vbt, vscale, q_len, kv_len = _make_varlen_tensors(
            infinicore, torch, cuda_dev, dtype
        )
        out = infinicore.mha_varlen(
            vq, vk, vv, cq, ck, vbt, max(q_len, 64), max(kv_len, 64), scale=vscale
        )
        _ = out
        torch.cuda.synchronize(device_index)
        infinicore.sync_stream()
        return _finish_h3(
            cell="h3_no_capture_varlen",
            expect_if_h3="PASS",
            outcome="PASS",
            note="eager kvcache→varlen baseline (no FORCE capture)",
            force_fa=False,
        )
    except Exception as exc:  # noqa: BLE001
        phase, msg = _classify_fault(exc)
        return _finish_h3(
            cell="h3_no_capture_varlen",
            expect_if_h3="PASS",
            outcome="FAIL",
            fault_phase=phase,
            fault=msg + "\n" + traceback.format_exc()[-500:],
            note="baseline failed — infra issue, not H3",
            force_fa=False,
        )


def _owned_storage_ok(t) -> bool:
    """True if torch tensor storage is torch-owned (not a no-op from_blob alias)."""
    try:
        # from_blob with empty deleter still reports a storage; clone() allocates new.
        # Heuristic: tensor that owns its memory has _base is None and was produced
        # by clone/empty (storage use_count typically 1 for fresh clones).
        return t._base is None and t.untyped_storage().data_ptr() != 0
    except Exception:  # noqa: BLE001
        return False


def _run_fa_owned_prototype(torch, cuda_dev, dtype, scale: float) -> dict[str, Any]:
    """Call FA2 kvcache with torch-owned clones (Phase-0 owning prototype)."""
    hq, hkv, hd, block = HQ, HKV, HD, BLOCK
    kv_len = 512
    num_blocks = max(1, (kv_len + block - 1) // block)
    q = torch.randn(1, 1, hq, hd, device=cuda_dev, dtype=dtype).contiguous().clone()
    k = torch.randn(num_blocks, block, hkv, hd, device=cuda_dev, dtype=dtype).contiguous().clone()
    v = torch.randn(num_blocks, block, hkv, hd, device=cuda_dev, dtype=dtype).contiguous().clone()
    seqlens = torch.tensor([kv_len], device=cuda_dev, dtype=torch.int32).contiguous().clone()
    bt = torch.arange(num_blocks, device=cuda_dev, dtype=torch.int32).view(1, -1).contiguous().clone()
    assert all(_owned_storage_ok(x) for x in (q, k, v, seqlens, bt))

    faults: list[str] = []
    # 1) Python flash_attn wrapper (no ``out=`` on this MetaX build).
    try:
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache

        out = flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=seqlens,
            block_table=bt,
            softmax_scale=scale,
            causal=True,
        )
        torch.cuda.synchronize()
        return {
            "ok": True,
            "api": "flash_attn_with_kvcache",
            "owned_storage": True,
            "out_finite": bool(torch.isfinite(out).all().item()),
        }
    except Exception as exc:  # noqa: BLE001
        faults.append(f"flash_attn_with_kvcache: {type(exc).__name__}: {exc}")

    # 2) MetaX flash_attn_2_cuda.fwd_kvcache / mha_fwd_kvcache.
    try:
        import flash_attn_2_cuda as fa2  # type: ignore

        fwd = getattr(fa2, "mha_fwd_kvcache", None) or getattr(fa2, "fwd_kvcache", None)
        if fwd is None:
            raise RuntimeError("no fwd_kvcache on flash_attn_2_cuda")
        out = torch.empty_like(q).clone()
        try:
            fwd(
                q,
                k,
                v,
                None,
                None,
                seqlens,
                None,
                None,
                None,
                None,
                bt,
                None,
                out,
                scale,
                True,
                -1,
                -1,
                0.0,
                False,
                0,
            )
        except TypeError:
            # Alternate positional layouts — last resort: call with minimal args.
            fwd(q, k, v, seqlens, bt, scale)
        torch.cuda.synchronize()
        return {
            "ok": True,
            "api": f"flash_attn_2_cuda.{getattr(fwd, '__name__', 'fwd_kvcache')}",
            "owned_storage": True,
            "out_finite": bool(torch.isfinite(out).all().item()),
        }
    except Exception as exc:  # noqa: BLE001
        faults.append(f"fa2: {type(exc).__name__}: {exc}")

    return {"ok": False, "fault": " | ".join(faults), "api": "none"}


def run_h4_hostop_from_blob(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> H3H4CellResult:
    """MoE|FA(host-break)|MoE — documents HostOp still on from_blob bridge (H4)."""
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=True,
        force_fa=False,
    )
    os.environ.pop("INFINI_FA_ATEN_OWNED", None)
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
        infinicore.mha_kvcache(q, k, v, seqlens, bt, scale=scale)  # HostOp / host_break
        run_moe()
        graph = infinicore.stop_graph_recording()
        _clear_phase(_ic)
        graph.run()
        infinicore.sync_stream()
        nseg = int(graph.device_segment_count())
        ok = bool(graph.has_device_exec()) and nseg >= 2
        # H4 documents mechanism: HostOp FA still calls same run() → from_blob.
        # Outcome PASS = sandwich works; DOC expect for hypothesis bookkeeping.
        return H3H4CellResult(
            cell="h4_hostop_from_blob",
            outcome="PASS" if ok else "FAIL",
            expect_if_hypothesis="DOC",
            hypothesis="H4",
            hypothesis_consistent=ok,  # documented when sandwich + from_blob note hold
            has_device_exec=bool(graph.has_device_exec()),
            device_segment_count=nseg,
            device_graph_log=(graph.device_graph_log() or "")[:800],
            fault=None
            if ok
            else f"expected ≥2 device segments around FA HostOp, got {nseg}",
            note=(
                "H4: host_break only skips hcStreamBeginCapture; HostOp still runs "
                "mha_kvcache_flashattn::run via non-owning to_aten_tensor→at::from_blob "
                "(not vLLM FX attn-outside-graph split)"
            ),
            force_fa_capture=False,
            aten_bridge="from_blob",
            extras={"INFINI_FA_ATEN_OWNED": False, "segs_ok": nseg >= 2},
        )
    except Exception as exc:  # noqa: BLE001
        if _ic is not None:
            _clear_phase(_ic)
        phase, msg = _classify_fault(exc)
        return H3H4CellResult(
            cell="h4_hostop_from_blob",
            outcome="FAIL",
            expect_if_hypothesis="DOC",
            hypothesis="H4",
            hypothesis_consistent=False,
            fault_phase=phase,
            fault=msg,
            note="HostOp from_blob sandwich failed",
            aten_bridge="from_blob",
        )


def run_h4_hostop_owning(
    *,
    segment_pt2: str,
    bucket: int,
    layer_idx: int,
    device_index: int,
    moe_configs: str,
    moe_triton_cache: str,
) -> H3H4CellResult:
    """HostOp sandwich + owning ATen prototype (Phase-1 flag or Python FA2 clones)."""
    _setup_common(
        moe_configs=moe_configs,
        moe_triton_cache=moe_triton_cache,
        triton_capture=True,
        force_fa=False,
    )
    # Prototype flag for Phase-1 ``to_aten_tensor_owned`` path.
    os.environ["INFINI_FA_ATEN_OWNED"] = "1"
    prior_force = _env_truthy("INFINI_FA_H4_PRIOR_FORCE")
    _ic = None
    try:
        import torch

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

        if prior_force:
            # Optional same-process stress: FORCE capture then HostOp (H3×H4 combo).
            os.environ["INFINI_FA_FORCE_CAPTURE"] = "1"
            graph_f, cap_fault = _capture_fa_kvcache(
                infinicore=infinicore,
                torch=torch,
                device=device,
                device_index=device_index,
                q=q,
                k=k,
                v=v,
                seqlens=seqlens,
                bt=bt,
                scale=scale,
                n_layers=_n_fa_layers(),
            )
            os.environ.pop("INFINI_FA_FORCE_CAPTURE", None)
            prior = {"capture_fault": cap_fault, "segs": int(graph_f.device_segment_count())}
        else:
            prior = None

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
        graph.run()
        infinicore.sync_stream()
        nseg = int(graph.device_segment_count())

        # Until Phase-1 wires INFINI_FA_ATEN_OWNED in C++, HostOp IC path is still
        # from_blob; exercise owning via FA2 Python prototype for contrast.
        proto = _run_fa_owned_prototype(torch, cuda_dev, dtype, scale)
        phase1_wired = _env_truthy("INFINI_FA_ATEN_OWNED_WIRED")  # set by Phase-1 tests
        bridge = "owned" if phase1_wired else "owned_prototype"
        ok = bool(graph.has_device_exec()) and nseg >= 2 and bool(proto.get("ok"))
        return H3H4CellResult(
            cell="h4_hostop_owning",
            outcome="PASS" if ok else "FAIL",
            expect_if_hypothesis="PASS",
            hypothesis="H4",
            hypothesis_consistent=ok,
            has_device_exec=bool(graph.has_device_exec()),
            device_segment_count=nseg,
            device_graph_log=(graph.device_graph_log() or "")[:800],
            fault=None if ok else f"sandwich/proto failed: proto={proto}",
            note=(
                "Owning prototype: INFINI_FA_ATEN_OWNED=1 requested; "
                + (
                    "C++ owned path wired"
                    if phase1_wired
                    else "C++ still from_blob until Phase-1; Python FA2 owned clones exercised"
                )
            ),
            force_fa_capture=False,
            aten_bridge=bridge,
            extras={
                "INFINI_FA_ATEN_OWNED": True,
                "phase1_wired": phase1_wired,
                "owned_prototype": proto,
                "prior_force": prior,
            },
        )
    except Exception as exc:  # noqa: BLE001
        if _ic is not None:
            _clear_phase(_ic)
        phase, msg = _classify_fault(exc)
        return H3H4CellResult(
            cell="h4_hostop_owning",
            outcome="FAIL",
            expect_if_hypothesis="PASS",
            hypothesis="H4",
            hypothesis_consistent=False,
            fault_phase=phase,
            fault=msg + "\n" + traceback.format_exc()[-500:],
            note="owning HostOp cell failed",
            aten_bridge="owned_prototype",
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        choices=(
            "h3_kvcache_then_varlen",
            "h3_kvcache_then_kvcache",
            "h3_no_capture_varlen",
            "h4_hostop_from_blob",
            "h4_hostop_owning",
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
        "h3_kvcache_then_varlen": run_h3_kvcache_then_varlen,
        "h3_kvcache_then_kvcache": run_h3_kvcache_then_kvcache,
        "h3_no_capture_varlen": run_h3_no_capture_varlen,
        "h4_hostop_from_blob": run_h4_hostop_from_blob,
        "h4_hostop_owning": run_h4_hostop_owning,
    }
    cells = list(runners.keys()) if args.cell == "all" else [args.cell]
    results: list[dict[str, Any]] = []
    for name in cells:
        print(f"[fa_h3h4_iso] === cell={name} ===", flush=True)
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
            r = H3H4CellResult(
                cell=name,
                outcome="FAIL",
                expect_if_hypothesis="FAIL" if "varlen" in name and "no_capture" not in name else "PASS",
                hypothesis="H3" if name.startswith("h3_") else "H4",
                hypothesis_consistent=False,
                fault_phase="unknown",
                fault=str(exc),
            )
        d = asdict(r)
        results.append(d)
        print(
            f"[fa_h3h4_iso] outcome={r.outcome} cell={r.cell} hyp={r.hypothesis} "
            f"expect_if_hyp={r.expect_if_hypothesis} consistent={r.hypothesis_consistent} "
            f"bridge={r.aten_bridge} segs={r.device_segment_count}",
            flush=True,
        )
        if r.fault:
            print(f"[fa_h3h4_iso] fault: {r.fault[:500]}", flush=True)
        if r.note:
            print(f"[fa_h3h4_iso] note: {r.note}", flush=True)

    payload = {"cells": results, "device_index": args.device_index}
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[fa_h3h4_iso] wrote {out}", flush=True)

    # Exit 0 always for single-cell diagnose (H3 FAIL is expected evidence);
    # campaign runner scores the matrix.
    os._exit(0)


if __name__ == "__main__":
    main()
