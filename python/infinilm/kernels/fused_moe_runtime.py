# Copyright (c) 2025, InfiniCore
# SPDX-License-Identifier: Apache-2.0
# Attribution: Triton MoE kernel adapted from vLLM fused_moe (Apache-2.0).
"""Tier-2 pinned FusedMoE launcher — serve loads cubins; never imports vLLM."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "infinilm.kernels.fused_moe_runtime requires triton (loader/launcher)"
    ) from exc

from infinilm.kernels.moe_guards import (  # noqa: E402
    allow_moe_jit,
    assert_no_vllm,
    launcher_hash,
    launcher_source_path,
)

# Back-compat aliases used inside this module.
_allow_jit = allow_moe_jit


def _configs_dir() -> Path:
    raw = os.environ.get("INFINI_MOE_CONFIGS", "").strip()
    if not raw:
        raise RuntimeError(
            "INFINI_MOE_CONFIGS is unset. Point it at moe_configs/ under the "
            "model hash dir (see docs/M5_moe_fused_artifact_design.md). "
            "Rebuild: ./scripts/rebuild_minicpm5_moe_artifacts.sh"
        )
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"INFINI_MOE_CONFIGS is not a directory: {path}")
    return path


def _triton_cache_dir() -> Path:
    raw = (
        os.environ.get("INFINI_MOE_TRITON_CACHE", "").strip()
        or os.environ.get("TRITON_CACHE_DIR", "").strip()
    )
    if not raw:
        raise RuntimeError(
            "INFINI_MOE_TRITON_CACHE / TRITON_CACHE_DIR unset. "
            "Rebuild: ./scripts/rebuild_minicpm5_moe_artifacts.sh"
        )
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"Triton MoE cache dir missing: {path}")
    os.environ["TRITON_CACHE_DIR"] = str(path)
    return path


def _device_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0).replace(" ", "_")


def _config_candidates(E: int, N: int, H: int) -> list[Path]:
    root = _configs_dir()
    dev = _device_name()
    aliases = [dev, "X203", "Mars_03"]
    names: list[str] = []
    for d in aliases:
        names.append(f"H={H},E={E},N={N},device_name={d}.json")
        names.append(f"E={E},N={N},device_name={d}.json")
    out: list[Path] = []
    for name in names:
        flat = root / name
        nested = root / f"H={H}" / name
        if flat not in out:
            out.append(flat)
        if nested not in out:
            out.append(nested)
    return out


@lru_cache(maxsize=8)
def _load_config_table(E: int, N: int, H: int) -> Dict[str, Any]:
    for path in _config_candidates(E, N, H):
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
    tried = ", ".join(str(p) for p in _config_candidates(E, N, H)[:6])
    raise RuntimeError(
        f"MoE Triton config JSON missing for E={E},N={N},H={H}. "
        f"Tried: {tried}. Rebuild moe_configs via rebuild_minicpm5_moe_artifacts.sh"
    )


def _flatten_stage(cfg: Any, stage: str) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        raise RuntimeError(f"invalid MoE config entry (expected dict): {cfg!r}")
    if stage in cfg and isinstance(cfg[stage], dict):
        return dict(cfg[stage])
    # Stock vLLM flat layout
    keys = ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M")
    if all(k in cfg for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")):
        out = {k: cfg[k] for k in keys if k in cfg}
        for opt in ("num_warps", "num_stages"):
            if opt in cfg:
                out[opt] = cfg[opt]
        return out
    raise RuntimeError(
        f"MoE config lacks {stage}/flat BLOCK_SIZE_* fields: keys={list(cfg)}"
    )


def get_moe_config_for_m(
    M: int, *, E: int, N: int, H: int, stage: str = "stage1"
) -> Dict[str, Any]:
    table = _load_config_table(E, N, H)
    keys = sorted(int(k) for k in table if str(k).isdigit())
    if not keys:
        raise RuntimeError("MoE config JSON has no numeric batch keys")
    # Largest key <= M; else smallest key
    le = [k for k in keys if k <= M]
    chosen = max(le) if le else keys[0]
    entry = table[str(chosen)]
    cfg = _flatten_stage(entry, stage)
    for req in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
        if req not in cfg:
            raise RuntimeError(f"MoE config for M~{chosen} missing {req}")
    cfg.setdefault("GROUP_SIZE_M", 1)
    return _sanitize_moe_config(cfg)


def _sanitize_moe_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp Triton tile/pipeline knobs to fit Mars (~64KiB) shared memory.

    Seeded MetaX configs for large M often use 128^3 + cpasync + num_stages=4,
    which requests >64KiB SMEM and fails at launch.
    """
    out = dict(cfg)
    bm = int(out["BLOCK_SIZE_M"])
    bn = int(out["BLOCK_SIZE_N"])
    bk = int(out["BLOCK_SIZE_K"])
    stages = int(out.get("num_stages", 2))
    # Conservative bf16 tile estimate (A+B) * stages; leave headroom.
    est = (bm * bk + bk * bn) * 2 * max(stages, 1)
    limit = int(os.environ.get("INFINI_MOE_SMEM_LIMIT", "65536"))
    if est <= limit and out.get("pipeline", "basic") in ("basic", "", None):
        return out
    # Fall back to a proven small-M tile (matches working M<=512 configs).
    out["BLOCK_SIZE_M"] = min(bm, 64)
    out["BLOCK_SIZE_N"] = min(bn, 64)
    out["BLOCK_SIZE_K"] = min(bk, 64)
    out["num_stages"] = min(stages, 3)
    out["num_warps"] = int(out.get("num_warps", 4))
    if out["num_warps"] > 8:
        out["num_warps"] = 8
    out["pipeline"] = "basic"
    out.pop("scenario", None)
    return out


# ---------------------------------------------------------------------------
# Triton kernels (Apache-2.0; adapted from vLLM fused_moe)
# ---------------------------------------------------------------------------


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def _fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + off_experts * stride_be + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _phase_marker(name: str) -> None:
    if os.environ.get("INFINI_MOE_PROFILE_PHASES", "").strip() in ("1", "true", "yes"):
        print(f"=== PHASE {name} ===", flush=True)


# #region agent log
_DEBUG_LOG_PATH = "/opt/offline/infinilm-metax-20260622/.cursor/debug-e5aec6.log"
_DEBUG_STATE = {"calls": 0}


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """NDJSON debug log for MoE host-break vs in-graph parity bisect."""
    try:
        import json
        import time

        payload = {
            "sessionId": "e5aec6",
            "runId": os.environ.get("INFINI_DEBUG_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


# #endregion


def _host_split_enabled() -> bool:
    return os.environ.get("INFINI_MOE_HOST_SPLIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class _HostSplitAccum:
    """Accumulate host-side ms for MoE opaque attribution (Phase 0)."""

    __slots__ = ("enabled", "totals", "_stack")

    def __init__(self) -> None:
        self.enabled = _host_split_enabled()
        self.totals: Dict[str, float] = {}
        self._stack: list = []

    def reset(self) -> None:
        self.totals.clear()
        self._stack.clear()

    def begin(self, name: str) -> None:
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time

        self._stack.append((name, time.perf_counter()))

    def end(self, name: str) -> None:
        if not self.enabled:
            return
        if not self._stack:
            return
        top_name, t0 = self._stack.pop()
        if top_name != name:
            # Mismatched begin/end — drop rather than corrupt totals.
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.totals[name] = self.totals.get(name, 0.0) + dt_ms

    def report(self, label: str = "host_split") -> Dict[str, float]:
        if not self.enabled:
            return {}
        parts = " ".join(f"{k}={v:.3f}" for k, v in sorted(self.totals.items()))
        total = sum(self.totals.values())
        print(
            f"[moe-host-split] {label} sum_keys_ms={total:.3f} {parts}",
            flush=True,
        )
        return dict(self.totals)


# Process-wide accumulator; profile harness resets between timed iters.
_HOST_SPLIT = _HostSplitAccum()


def host_split_reset() -> None:
    _HOST_SPLIT.reset()
    _HOST_SPLIT.enabled = _host_split_enabled()


def host_split_report(label: str = "host_split") -> Dict[str, float]:
    return _HOST_SPLIT.report(label)


def _moe_capture_safe_enabled() -> bool:
    """INFINI_MOE_CAPTURE_SAFE=1: aten MoE under stream capture; Triton eager."""
    raw = os.environ.get("INFINI_MOE_CAPTURE_SAFE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _moe_triton_capture_enabled() -> bool:
    """Whether Triton fused_moe_routed may run under stream capture.

    Defers to InfiniCore ``moe_triton_capture_allowed`` (FORCE_CAPTURE /
    deprecated TRITON_CAPTURE alias + Decode; MetaX needs METAX_CAPTURE_UNSAFE).
    ``INFINI_MOE_FORCE_HOST_BREAK=1`` forces off.
    """
    raw_force = os.environ.get("INFINI_MOE_FORCE_HOST_BREAK", "").strip().lower()
    if raw_force in ("1", "true", "yes", "on"):
        return False
    if os.environ.get("INFINI_MOE_TRITON_CAPTURE"):
        import warnings

        warnings.warn(
            "INFINI_MOE_TRITON_CAPTURE is deprecated; treat truthy as "
            "INFINI_MOE_FORCE_CAPTURE (Decode-only). MetaX also needs "
            "INFINI_MOE_METAX_CAPTURE_UNSAFE=1 (MoE-in-graph garbles by default)",
            DeprecationWarning,
            stacklevel=2,
        )
        if os.environ.get("INFINI_MOE_TRITON_CAPTURE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            os.environ.setdefault("INFINI_MOE_FORCE_CAPTURE", "1")
    try:
        from infinicore.lib import _infinicore as _ic

        if hasattr(_ic, "moe_triton_capture_allowed"):
            return bool(_ic.moe_triton_capture_allowed())
    except Exception:  # noqa: BLE001
        pass
    # Fallback when C++ binding unavailable: non-eager policy only
    # (phase is C++ TLS; under capture Decode guard is expected).
    policy = os.environ.get("INFINI_CUDAGRAPH_POLICY", "eager").strip().lower()
    return policy not in ("", "eager") and _under_device_stream_capture()


def _under_device_stream_capture() -> bool:
    """True while InfiniCore is inside hcStreamBeginCapture..EndCapture.

    Prefer the pybind TLS flag; also read ``INFINI_DEVICE_STREAM_CAPTURING`` via
    libc ``getenv`` (not ``os.environ``) — C++ ``setenv`` bridges duplicate-TLS /
    separate-DSO cases (same pattern as Infiniop), and ``os.environ`` can miss it.
    """
    try:
        from infinicore.lib import _infinicore as _ic

        if bool(_ic.is_device_stream_capturing()):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        import ctypes

        getenv = ctypes.CDLL(None).getenv
        getenv.argtypes = [ctypes.c_char_p]
        getenv.restype = ctypes.c_char_p
        raw = getenv(b"INFINI_DEVICE_STREAM_CAPTURING")
        if raw:
            v = raw.decode("utf-8", "replace").strip().lower()
            return v in ("1", "true", "yes", "on")
    except Exception:  # noqa: BLE001
        pass
    return False


def _empty_capture(numel: int, *, dtype, device) -> torch.Tensor:
    """IC-backed empty under CaptureArena; else torch.empty."""
    if _under_device_stream_capture():
        from infinicore.lib import _infinicore as _ic

        proto = torch.empty(0, dtype=dtype, device=device)
        return _ic.capture_empty_like(proto, [int(numel)])
    return torch.empty(int(numel), dtype=dtype, device=device)


def _zero_capture(t: torch.Tensor) -> None:
    """Capture-safe zero: MetaX ATen ``zero_()`` / ``FillFunctor`` HTC's on IC arena temps.

    Under device-stream capture, call ``hcMemsetAsync`` on the current capture stream
    (Memcpy/memset graph node) instead of ``vectorized_elementwise_kernel`` FillFunctor.
    """
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return
    if not _under_device_stream_capture():
        t.zero_()
        return
    if not t.is_contiguous():
        # Rare under capture; fall back to H2D of a contiguous host buffer.
        host = torch.zeros(t.shape, dtype=t.dtype, device="cpu").contiguous()
        _retain_capture(host)
        t.copy_(host, non_blocking=False)
        return
    import ctypes

    lib = ctypes.CDLL("/opt/hpcc/lib/libhcruntime.so")
    # hcError_t hcMemsetAsync(void *devPtr, int value, size_t count, hcStream_t stream)
    lib.hcMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.hcMemsetAsync.restype = ctypes.c_int
    stream = torch.cuda.current_stream().cuda_stream
    nbytes = int(t.numel() * t.element_size())
    st = lib.hcMemsetAsync(ctypes.c_void_p(t.data_ptr()), 0, nbytes, ctypes.c_void_p(stream))
    if st != 0:
        raise RuntimeError(f"_zero_capture: hcMemsetAsync failed status={st}")


def _fill_capture(t: torch.Tensor, value: int) -> None:
    """Capture-safe scalar fill via host H2D (hcMemset only supports 0-byte pattern).

    Always ``non_blocking=False`` under capture so H2D stays on the capture stream
    (unjoined async H2D can invalidate MetaX ``hcStreamEndCapture``).

    Host staging tensors are retained on the CaptureArena so MetaX graph replay does
    not HTC on a freed host pointer (ephemeral CPU buffer would dangle on probe).
    """
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return
    if not _under_device_stream_capture():
        t.fill_(value)
        return
    if value == 0:
        _zero_capture(t)
        return
    host = torch.full(t.shape, value, dtype=t.dtype, device="cpu").contiguous()
    _retain_capture(host)
    t.copy_(host, non_blocking=False)


def _retain_capture(*tensors) -> None:
    """Retain residual torch-allocator temps on the active CaptureArena."""
    if not _under_device_stream_capture():
        return
    try:
        from infinicore.lib import _infinicore as _ic

        retain = getattr(_ic, "capture_retain", None)
        if retain is None:
            return
        for t in tensors:
            if isinstance(t, torch.Tensor):
                retain(t)
    except Exception:  # noqa: BLE001
        return


# Persistent decode-sized aten workspaces: same device addresses every call so
# MetaX FULL capture can record MoE even when Python TLS/env miss capturing
# (duplicate libinfinicore TLS → CaptureArena invisible to Python).
_ATEN_STATIC_WS: Dict[tuple, dict] = {}


def _zero_device_buffer(t: torch.Tensor) -> None:
    """Zero a device tensor via hcMemsetAsync when possible (capture-safe)."""
    if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
        return
    if not t.is_cuda:
        t.zero_()
        return
    try:
        import ctypes

        lib = ctypes.CDLL("libhcrt.so")
        lib.hcMemsetAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.hcMemsetAsync.restype = ctypes.c_int
        stream = torch.cuda.current_stream().cuda_stream
        nbytes = int(t.numel() * t.element_size())
        st = lib.hcMemsetAsync(
            ctypes.c_void_p(t.data_ptr()), 0, nbytes, ctypes.c_void_p(stream)
        )
        if st == 0:
            return
    except Exception:  # noqa: BLE001
        pass
    t.zero_()


def _routed_experts_aten_static(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Decode-sized aten MoE into process-lifetime buffers (CG address stability)."""
    top_k = int(topk_ids.shape[1])
    t_tokens = int(x.size(0))
    hidden = int(x.size(1))
    n2 = int(w_gate_up.size(1))
    key = (str(x.device), str(x.dtype), t_tokens, hidden, top_k, n2)
    ws = _ATEN_STATIC_WS.get(key)
    if ws is None:
        ws = {
            "acc": torch.empty((t_tokens, hidden), dtype=x.dtype, device=x.device),
            "gu": torch.empty((t_tokens, n2), dtype=x.dtype, device=x.device),
            "h": torch.empty((t_tokens, n2 // 2), dtype=x.dtype, device=x.device),
            "y": torch.empty((t_tokens, hidden), dtype=x.dtype, device=x.device),
            "w": torch.empty((t_tokens, 1), dtype=x.dtype, device=x.device),
            "x3": torch.empty((t_tokens, hidden, 1), dtype=x.dtype, device=x.device),
            "h3": torch.empty((t_tokens, n2 // 2, 1), dtype=x.dtype, device=x.device),
        }
        _ATEN_STATIC_WS[key] = ws
        # #region agent log
        _agent_log(
            "H8",
            "fused_moe_runtime.py:_routed_experts_aten_static",
            "aten_static_ws_alloc",
            {"key": list(key), "top_k": top_k},
        )
        # #endregion
    acc = ws["acc"]
    _zero_device_buffer(acc)
    # Keep a stable view of x for bmm without per-call unsqueeze alloc when possible.
    x3 = ws["x3"]
    x3.copy_(x.unsqueeze(-1))
    for k in range(top_k):
        idx = topk_ids[:, k]
        w_gu = w_gate_up.index_select(0, idx)
        w_d = w_down.index_select(0, idx)
        gu = ws["gu"]
        gu3 = gu.view(t_tokens, n2, 1)
        torch.bmm(w_gu, x3, out=gu3)
        gate, up = gu.chunk(2, dim=-1)
        # Fresh silu each expert; mul into static h (avoid ephemeral silu*up).
        silu_g = F.silu(gate)
        h = ws["h"]
        torch.mul(silu_g, up, out=h)
        h3 = ws["h3"]
        h3.copy_(h.unsqueeze(-1))
        y = ws["y"]
        y3 = y.view(t_tokens, hidden, 1)
        torch.bmm(w_d, h3, out=y3)
        w = ws["w"]
        wk = topk_weights[:, k].unsqueeze(-1)
        if wk.dtype != x.dtype:
            w.copy_(wk.to(dtype=x.dtype))
        else:
            w.copy_(wk)
        acc.add_(y * w)
    return acc


def _routed_experts_aten(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Routed experts via index_select + bmm (no Triton).

    Same shapes as ``fused_moe_routed``. Under capture, retain Torch temps so
    MetaX replay does not dangle (parity with host-break freshness).

    When ``INFINI_MOE_FORCE_CAPTURE`` + decode-sized T and Python cannot see
    CaptureArena (TLS split), use process-static buffers (H8) so CG replay
    keeps stable addresses — ``torch.zeros_like`` under a live capture garbles.
    """
    top_k = int(topk_ids.shape[1])
    capturing = _under_device_stream_capture()
    force_cap = os.environ.get("INFINI_MOE_FORCE_CAPTURE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    decode_sized = int(x.size(0)) <= 16
    # Prefer arena when TLS works; else static workspaces under FORCE_CAPTURE.
    if force_cap and decode_sized and not capturing:
        return _routed_experts_aten_static(
            x, topk_weights, topk_ids, w_gate_up, w_down
        )
    if capturing:
        try:
            acc = _empty_capture(int(x.numel()), dtype=x.dtype, device=x.device).view(
                x.shape
            )
            _zero_capture(acc)
        except Exception:  # noqa: BLE001
            return _routed_experts_aten_static(
                x, topk_weights, topk_ids, w_gate_up, w_down
            )
    else:
        acc = torch.zeros_like(x)
    retained: list = []
    for k in range(top_k):
        idx = topk_ids[:, k]
        w_gu = w_gate_up.index_select(0, idx)
        w_d = w_down.index_select(0, idx)
        gu = torch.bmm(w_gu, x.unsqueeze(-1)).squeeze(-1)
        gate, up = gu.chunk(2, dim=-1)
        silu_g = F.silu(gate)
        if capturing:
            h = _empty_capture(
                int(gate.numel()), dtype=gate.dtype, device=gate.device
            ).view_as(gate)
            torch.mul(silu_g, up, out=h)
            retained.extend((w_gu, w_d, gu, silu_g, h))
        else:
            h = silu_g * up
        y = torch.bmm(w_d, h.unsqueeze(-1)).squeeze(-1)
        w = topk_weights[:, k].unsqueeze(-1)
        if w.dtype != x.dtype:
            w = _capture_safe_to_dtype(w, x.dtype) if capturing else w.to(dtype=x.dtype)
        if capturing:
            contrib = y * w
            retained.extend((y, contrib, w))
            acc.add_(contrib)
        else:
            acc = acc + y * w
    if capturing:
        _retain_capture(acc, *retained)
    return acc


def _moe_align_block_size_host(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU align for small ``numel`` (decode M=1): one D2H + H2D, no GPU kernel storm."""
    import numpy as np

    device = topk_ids.device
    flat_t = topk_ids.reshape(-1)
    numel = int(flat_t.numel())

    if numel == 0:
        max_num_tokens_padded = num_experts * (block_size - 1)
        sorted_ids = torch.full(
            (max(max_num_tokens_padded, 0),), 0, dtype=torch.int32, device=device
        )
        expert_ids_out = torch.full((1,), -1, dtype=torch.int32, device=device)
        num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=device)
        return sorted_ids, expert_ids_out, num_tokens_post_pad

    # Single small D2H (replaces many MetaX kernel launches + former ``.item()`` syncs).
    flat = flat_t.detach().to(dtype=torch.int64, device="cpu").numpy()
    # CG warmup / zeroed capture inputs can yield out-of-range or negative expert
    # ids; clamp so ``np.bincount`` does not raise during instantiate warmup.
    if num_experts > 0:
        flat = np.clip(flat, 0, num_experts - 1)
    order = np.argsort(flat, kind="stable")
    sorted_experts = flat[order]

    counts = np.bincount(sorted_experts, minlength=num_experts).astype(np.int64)
    rem = counts % block_size
    padded_counts = counts + np.where(rem == 0, 0, block_size - rem)
    n_post = int(padded_counts.sum())

    padded_offsets = np.zeros(num_experts + 1, dtype=np.int64)
    padded_offsets[1:] = np.cumsum(padded_counts)

    first_pos = np.full(num_experts, numel, dtype=np.int64)
    # First occurrence of each expert in sorted order.
    for i, e in enumerate(sorted_experts):
        if first_pos[e] == numel:
            first_pos[e] = i

    idx = np.arange(numel, dtype=np.int64)
    within_expert = idx - first_pos[sorted_experts]
    out_pos = padded_offsets[sorted_experts] + within_expert

    # Exact ``n_post`` buffer (not max-pad) — smaller H2D + tighter Triton grid.
    sorted_ids_np = np.full(n_post, numel, dtype=np.int32)
    sorted_ids_np[out_pos] = order.astype(np.int32)

    n_m_blocks = max(n_post // block_size, 1)
    expert_ids_np = np.full(n_m_blocks, -1, dtype=np.int32)
    block_idx = out_pos // block_size
    expert_ids_np[block_idx] = sorted_experts.astype(np.int32)

    sorted_ids = torch.as_tensor(sorted_ids_np, dtype=torch.int32, device=device)
    expert_ids_out = torch.as_tensor(expert_ids_np, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.tensor([n_post], dtype=torch.int32, device=device)
    return sorted_ids, expert_ids_out, num_tokens_post_pad


def _capture_safe_to_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Capture-safe dtype cast: avoid ATen ``.to`` FillFunctor on MetaX under hcStream capture.

    Prefer CaptureArena empty + ``copy_`` (same pattern as C++ ``capture_safe_to_dtype``).
    """
    if t is None or not isinstance(t, torch.Tensor) or t.dtype == dtype:
        return t
    if not _under_device_stream_capture():
        return t.to(dtype)
    out = _empty_capture(int(t.numel()), dtype=dtype, device=t.device)
    if tuple(t.shape) != (t.numel(),):
        out = out.view(t.shape)
    out.copy_(t)
    _retain_capture(out)
    return out


def _moe_align_block_size_capture(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device align under hcStream capture — arena temps, no ``torch.bincount``/``argsort``.

    Decode-sized path uses an O(n²) counting-sort placement (no ATen ``argsort`` /
    ``FillFunctor<long>`` scratch) so MetaX graph replay does not ATU. Host D2H of
    ids under capture is unsafe on MetaX (garbage reads).
    """
    device = topk_ids.device
    numel = int(topk_ids.numel())

    if numel == 0:
        max_num_tokens_padded = num_experts * (block_size - 1)
        sorted_ids = _empty_capture(max(max_num_tokens_padded, 0), dtype=torch.int32, device=device)
        _zero_capture(sorted_ids)
        expert_ids_out = _empty_capture(1, dtype=torch.int32, device=device)
        _fill_capture(expert_ids_out, -1)
        num_tokens_post_pad = _empty_capture(1, dtype=torch.int32, device=device)
        _zero_capture(num_tokens_post_pad)
        return sorted_ids, expert_ids_out, num_tokens_post_pad

    flat_i32 = topk_ids.reshape(-1)
    if flat_i32.dtype != torch.int32:
        flat_i32 = _capture_safe_to_dtype(flat_i32, torch.int32)
    # Scatter index API wants Long — arena cast (no ``.to(int64)`` FillFunctor).
    flat = _capture_safe_to_dtype(flat_i32, torch.int64)
    idx = _empty_capture(numel, dtype=torch.int64, device=device)
    host_idx = torch.arange(numel, dtype=torch.int64, device="cpu").contiguous()
    _retain_capture(host_idx)
    idx.copy_(host_idx, non_blocking=False)

    counts = _empty_capture(num_experts, dtype=torch.int64, device=device)
    _zero_capture(counts)
    ones = _empty_capture(numel, dtype=torch.int64, device=device)
    _fill_capture(ones, 1)
    counts.scatter_add_(0, flat, ones)

    rem = counts % block_size
    pad = (block_size - rem) % block_size
    padded_counts = counts + pad
    num_tokens_post_pad = _capture_safe_to_dtype(padded_counts.sum().reshape(1), torch.int32)

    padded_offsets = _empty_capture(num_experts + 1, dtype=torch.int64, device=device)
    _zero_capture(padded_offsets)
    padded_offsets_tail = padded_counts.cumsum(0)
    padded_offsets[1:] = padded_offsets_tail

    # within[i] = #{j < i | flat[j]==flat[i]} via lower-triangular eq sum (no argsort).
    fi = flat.unsqueeze(0).expand(numel, numel)
    fj = flat.unsqueeze(1).expand(numel, numel)
    j_idx = idx.unsqueeze(0).expand(numel, numel)
    i_idx = idx.unsqueeze(1).expand(numel, numel)
    # Avoid ``torch.where`` under capture — MetaX records Fill/where nodes that ATU on
    # probe/replay. Bool compare + sum is enough for the O(n²) within-count.
    within = ((fi == fj) & (j_idx < i_idx)).sum(dim=1)

    out_pos = padded_offsets[flat] + within
    if numel <= 64:
        max_num_tokens_padded = numel + numel * (block_size - 1)
    else:
        max_num_tokens_padded = numel + num_experts * (block_size - 1)
    n_m_blocks = max(max_num_tokens_padded // max(block_size, 1), 1)

    sorted_ids = _empty_capture(max_num_tokens_padded, dtype=torch.int32, device=device)
    _fill_capture(sorted_ids, numel)
    order_i32 = _capture_safe_to_dtype(idx, torch.int32)
    sorted_ids.scatter_(0, out_pos, order_i32)

    expert_ids_out = _empty_capture(n_m_blocks, dtype=torch.int32, device=device)
    _fill_capture(expert_ids_out, -1)
    block_idx = out_pos // block_size
    flat_i32_out = _capture_safe_to_dtype(flat, torch.int32)
    expert_ids_out.scatter_(0, block_idx, flat_i32_out)

    _retain_capture(
        flat,
        flat_i32,
        idx,
        ones,
        counts,
        rem,
        pad,
        padded_counts,
        num_tokens_post_pad,
        padded_offsets,
        padded_offsets_tail,
        fi,
        fj,
        j_idx,
        i_idx,
        within,
        out_pos,
        sorted_ids,
        expert_ids_out,
        block_idx,
        order_i32,
        flat_i32_out,
    )
    return sorted_ids, expert_ids_out, num_tokens_post_pad


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized align (no vLLM ``_custom_ops``, no mid-align ``.item()`` syncs).

    For small token counts (decode ``M=1``, ``TOP_K=16`` → 16 ids) uses a host
    path: one D2H of ids + CPU align + H2D of outputs — avoids MetaX launch tax
    from argsort/bincount/scatter storms. Larger ``numel`` stays on-device with
    pad-sentinel init (no redundant pad_fill / ``repeat_interleave``).
    """
    device = topk_ids.device
    numel = int(topk_ids.numel())
    # Under hcStream capture: arena-backed path (no host D2H, no bincount).
    if _under_device_stream_capture():
        _HOST_SPLIT.begin("align_capture")
        out = _moe_align_block_size_capture(topk_ids, block_size, num_experts)
        _HOST_SPLIT.end("align_capture")
        # #region agent log
        if _DEBUG_STATE["calls"] < 8:
            _agent_log(
                "H1",
                "fused_moe_runtime.py:moe_align_block_size",
                "align_path",
                {
                    "path": "align_capture",
                    "numel": numel,
                    "block_size": int(block_size),
                    "num_experts": int(num_experts),
                    "sorted_len": int(out[0].numel()),
                    "expert_len": int(out[1].numel()),
                },
            )
        # #endregion
        return out
    # Decode-sized (e.g. M=1,TOP_K=16 → 16 ids): host path wins on MetaX.
    # Keep larger prefill-like aligns on-device (avoid big D2H).
    if numel > 0 and numel <= 64 and topk_ids.is_cuda:
        _HOST_SPLIT.begin("align_host_small")
        out = _moe_align_block_size_host(topk_ids, block_size, num_experts)
        _HOST_SPLIT.end("align_host_small")
        # #region agent log
        if _DEBUG_STATE["calls"] < 8:
            _agent_log(
                "H1",
                "fused_moe_runtime.py:moe_align_block_size",
                "align_path",
                {
                    "path": "align_host_small",
                    "numel": numel,
                    "block_size": int(block_size),
                    "num_experts": int(num_experts),
                    "sorted_len": int(out[0].numel()),
                    "expert_len": int(out[1].numel()),
                },
            )
        # #endregion
        return out

    flat = topk_ids.reshape(-1).to(torch.int64)
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    # Host-known: #blocks with ≥1 real token ≤ numel (ceil(c/B) ≤ c).
    n_m_blocks = max(numel, 1)

    if numel == 0:
        sorted_ids = torch.full(
            (max_num_tokens_padded,), 0, dtype=torch.int32, device=device
        )
        expert_ids_out = torch.full((n_m_blocks,), -1, dtype=torch.int32, device=device)
        num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=device)
        return sorted_ids, expert_ids_out, num_tokens_post_pad

    _HOST_SPLIT.begin("align_pre_item")
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    idx = torch.arange(numel, device=device, dtype=torch.int64)

    counts = torch.bincount(sorted_experts, minlength=num_experts)
    padded_counts = counts + (block_size - counts % block_size) % block_size
    # Stay on device — Triton loads this scalar; no ``.item()``.
    num_tokens_post_pad = padded_counts.sum().to(dtype=torch.int32).reshape(1)

    padded_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    padded_offsets[1:] = padded_counts.cumsum(0)

    first_pos = torch.full((num_experts,), numel, dtype=torch.int64, device=device)
    first_pos.scatter_reduce_(0, sorted_experts, idx, reduce="amin", include_self=True)

    within_expert = idx - first_pos[sorted_experts]
    out_pos = padded_offsets[sorted_experts] + within_expert

    # Init to pad sentinel; real tokens overwrite via scatter. Trailing pad
    # slots in each expert region remain ``numel`` — no explicit pad_fill.
    sorted_ids = torch.full((max_num_tokens_padded,), numel, dtype=torch.int32, device=device)
    sorted_ids.scatter_(0, out_pos, order.to(torch.int32))

    expert_ids_out = torch.full((n_m_blocks,), -1, dtype=torch.int32, device=device)
    block_idx = out_pos // block_size
    expert_ids_out.scatter_(0, block_idx, sorted_experts.to(torch.int32))
    _HOST_SPLIT.end("align_pre_item")
    return sorted_ids, expert_ids_out, num_tokens_post_pad


def moe_sum(input_3d: torch.Tensor, output: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reduce ``[M, topk, H] -> [M, H]`` (vLLM ``moe_sum`` semantics)."""
    if input_3d.dim() != 3:
        raise ValueError(f"moe_sum expects [M,K,H], got {tuple(input_3d.shape)}")
    # Prefer InfiniCore when available; else Torch (same numerics for bf16 sum).
    try:
        from infinicore.ops.moe_sum import moe_sum as _ic_moe_sum
        from infinicore.tensor import Tensor as ICTensor

        # Torch path is the serve default for ATen tensors from AOTI/custom op.
        del _ic_moe_sum, ICTensor
    except Exception:
        pass
    reduced = input_3d.sum(dim=1)
    if output is not None:
        output.copy_(reduced)
        return output
    return reduced


def _invoke_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: Any,
) -> None:
    cache_dir = _triton_cache_dir()
    if not _allow_jit():
        # Serve: refuse empty cache (cubin must already exist from warmup).
        if not any(cache_dir.iterdir()):
            raise RuntimeError(
                f"MoE Triton cache empty at {cache_dir} and INFINI_MOE_ALLOW_JIT=0. "
                "Run ./scripts/warmup_fused_moe_triton_cache.sh then pack artifacts."
            )

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(
            sorted_token_ids.size(0),
            A.size(0) * top_k * config["BLOCK_SIZE_M"],
        )
    # #region agent log
    if _DEBUG_STATE["calls"] <= 6:
        _agent_log(
            "H1",
            "fused_moe_runtime.py:_invoke_kernel",
            "triton_grid",
            {
                "call": _DEBUG_STATE["calls"],
                "capturing": _under_device_stream_capture(),
                "EM": int(EM),
                "sorted_len": int(sorted_token_ids.size(0)),
                "A0": int(A.size(0)),
                "top_k": int(top_k),
                "BLOCK_M": int(config["BLOCK_SIZE_M"]),
                "mul_routed": bool(mul_routed_weight),
            },
        )
    # #endregion
    grid = lambda META: (  # noqa: E731
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    kwargs = {
        "BLOCK_SIZE_M": int(config["BLOCK_SIZE_M"]),
        "BLOCK_SIZE_N": int(config["BLOCK_SIZE_N"]),
        "BLOCK_SIZE_K": int(config["BLOCK_SIZE_K"]),
        "GROUP_SIZE_M": int(config.get("GROUP_SIZE_M", 1)),
    }
    for opt in ("num_warps", "num_stages"):
        if opt in config:
            kwargs[opt] = int(config[opt])

    try:
        _fused_moe_kernel[grid](
            A,
            B,
            C,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.size(1),
            B.size(2),
            EM,
            A.size(0) * top_k,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            **kwargs,
        )
    except Exception as exc:
        if not _allow_jit():
            raise RuntimeError(
                "FusedMoE Triton launch failed with JIT disabled. "
                f"Cache={cache_dir}. Rebuild: ./scripts/rebuild_minicpm5_moe_artifacts.sh. "
                f"Underlying: {exc}"
            ) from exc
        raise


class _RoutedWorkspace:
    """Persistent scratch for ``fused_moe_routed`` (Phase 3: stop per-call alloc)."""

    __slots__ = ("cache13", "cache2", "out", "device", "dtype", "cap13", "cap2", "cap_out")

    def __init__(self) -> None:
        self.cache13: Optional[torch.Tensor] = None
        self.cache2: Optional[torch.Tensor] = None
        self.out: Optional[torch.Tensor] = None
        self.device = None
        self.dtype = None
        self.cap13 = 0
        self.cap2 = 0
        self.cap_out = 0

    def acquire(
        self,
        *,
        num_tokens: int,
        top_k: int,
        N2: int,
        H: int,
        N: int,
        device,
        dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        need13 = num_tokens * top_k * max(N2, H)
        need2 = num_tokens * top_k * N
        need_out = num_tokens * H
        # Under hcStream capture: always allocate IC-backed temps (do not reuse
        # torch.empty caches from pre-capture warmup — those are not arena-owned).
        if _under_device_stream_capture():
            cache13 = _empty_capture(need13, dtype=dtype, device=device)
            cache2 = _empty_capture(need2, dtype=dtype, device=device)
            out_buf = _empty_capture(need_out, dtype=dtype, device=device)
            intermediate_cache1 = cache13[: num_tokens * top_k * N2].view(
                num_tokens, top_k, N2
            )
            intermediate_cache3 = cache13[: num_tokens * top_k * H].view(
                num_tokens, top_k, H
            )
            intermediate_cache2 = cache2[:need2].view(num_tokens * top_k, N)
            out = out_buf[:need_out].view(num_tokens, H)
            return intermediate_cache1, intermediate_cache2, intermediate_cache3, out
        if (
            self.cache13 is None
            or self.device != device
            or self.dtype != dtype
            or self.cap13 < need13
        ):
            self.cache13 = torch.empty(need13, device=device, dtype=dtype)
            self.cap13 = need13
            self.device = device
            self.dtype = dtype
        if self.cache2 is None or self.cap2 < need2 or self.cache2.device != device:
            self.cache2 = torch.empty(need2, device=device, dtype=dtype)
            self.cap2 = need2
        if self.out is None or self.cap_out < need_out or self.out.device != device:
            self.out = torch.empty(need_out, device=device, dtype=dtype)
            self.cap_out = need_out
        intermediate_cache1 = self.cache13[: num_tokens * top_k * N2].view(
            num_tokens, top_k, N2
        )
        intermediate_cache3 = self.cache13[: num_tokens * top_k * H].view(
            num_tokens, top_k, H
        )
        intermediate_cache2 = self.cache2[:need2].view(num_tokens * top_k, N)
        out = self.out[:need_out].view(num_tokens, H)
        return intermediate_cache1, intermediate_cache2, intermediate_cache3, out


_ROUTED_WS = _RoutedWorkspace()


def fused_moe_routed(
    x: torch.Tensor,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Unquantized bf16/fp16 routed experts (Tier-2 entrypoint).

    x: [T, H]; topk_*: [T, K]; w_gate_up [E, 2I, H]; w_down [E, H, I].

    Capture modes (under ``hcStreamBeginCapture``):
    - Decode-sized (``T<=16``) + capture allowed: aten index_select+bmm —
      parity with host-break numerics / fresh buffers (Triton-under-capture still
      garbles Gate C Cell B). Bucket/prefill capture (``T>16``) keeps Triton +
      ``align_capture`` (aten OOMs on T=512).
    - ``INFINI_MOE_CAPTURE_SAFE=1`` (any T, Triton-capture off): aten.
    - ``INFINI_MOE_FORCE_HOST_BREAK=1``: force host-break (bisect).
    Otherwise (eager / host-break): Triton + host_align for small numel.
    """
    assert_no_vllm()
    if not x.is_cuda:
        raise RuntimeError("fused_moe_routed requires CUDA/PrivateUse1 tensors")
    assert x.dim() == 2
    assert topk_w.shape == topk_ids.shape
    assert w_gate_up.dim() == 3 and w_down.dim() == 3
    assert x.size(1) == w_gate_up.size(2) == w_down.size(1)

    _capturing = _under_device_stream_capture()
    _triton_cap = _moe_triton_capture_enabled()
    _safe = _moe_capture_safe_enabled()
    _t_tokens = int(x.size(0))
    # Host-break parity for decode-sized MoE body (aten index_select+bmm).
    # Do NOT rely solely on pybind ``moe_triton_capture_allowed`` / TLS phase:
    # InfiniLM C++ may set InferencePhase::Decode on a different libinfinicore
    # TLS than the Python extension, so Gate C logs showed triton_cap=False
    # while C++ still folded MoE in-graph (segs=1) → Triton-under-capture garble.
    _force_hb = os.environ.get("INFINI_MOE_FORCE_HOST_BREAK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _force_cap = os.environ.get("INFINI_MOE_FORCE_CAPTURE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    _decode_sized = _t_tokens <= 16
    # Aten body only under live capture / CAPTURE_SAFE / C++ triton_cap.
    # Do NOT key off FORCE_CAPTURE alone: on MetaX MoE stays host-break unless
    # METAX_CAPTURE_UNSAFE, and FORCE_CAPTURE+aten during HB skipped host-topk
    # D2H sync → race (post-fix19 all-118 garble).
    _use_aten_capture = (not _force_hb) and (
        (_capturing and _safe and not _triton_cap)
        or (_triton_cap and _decode_sized)
        or (_capturing and _decode_sized)
    )
    # #region agent log
    _DEBUG_STATE["calls"] = int(_DEBUG_STATE["calls"]) + 1
    _call_n = int(_DEBUG_STATE["calls"])
    _env_cap = None
    try:
        import ctypes

        _g = ctypes.CDLL(None).getenv
        _g.argtypes = [ctypes.c_char_p]
        _g.restype = ctypes.c_char_p
        _raw = _g(b"INFINI_DEVICE_STREAM_CAPTURING")
        _env_cap = _raw.decode() if _raw else ""
    except Exception:  # noqa: BLE001
        _env_cap = "err"
    _env_cap_on = str(_env_cap).strip().lower() in ("1", "true", "yes", "on")
    # Separate budgets: eager noise vs capture/force_capture (critical path).
    _crit = bool(_capturing or _env_cap_on or _force_cap)
    _log_budget_key = "logged_crit" if _crit else "logged"
    _log_limit = 128 if _crit else 48
    _log_this = _crit or _call_n <= 8 or (_decode_sized and _call_n <= 64)
    if _log_this and int(_DEBUG_STATE.get(_log_budget_key, 0)) < _log_limit:
        _DEBUG_STATE[_log_budget_key] = int(_DEBUG_STATE.get(_log_budget_key, 0)) + 1
        body = (
            "aten_parity_decode"
            if _use_aten_capture
            else ("triton_capture" if (_capturing or _env_cap_on) else "triton_eager_or_hb")
        )
        _agent_log(
            "H6",
            "fused_moe_runtime.py:fused_moe_routed",
            "moe_body_path",
            {
                "call": _call_n,
                "capturing": _capturing,
                "env_DEVICE_STREAM_CAPTURING": _env_cap,
                "triton_capture_allowed": _triton_cap,
                "force_capture_env": _force_cap,
                "capture_safe": _safe,
                "body": body,
                "T": _t_tokens,
                "K": int(topk_ids.size(1)),
                "E": int(w_gate_up.size(0)),
            },
        )
    # #endregion
    if _use_aten_capture:
        return _routed_experts_aten(x, topk_w, topk_ids, w_gate_up, w_down)

    num_tokens = x.size(0)
    E, N2, H = w_gate_up.shape
    N = N2 // 2
    K_dim = w_down.size(1)
    assert K_dim == H
    I = w_down.size(2)
    assert N == I
    top_k = int(topk_ids.size(1))

    if x.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif x.dtype == torch.float16:
        compute_type = tl.float16
    else:
        raise ValueError(f"unsupported dtype {x.dtype}; use bf16/fp16")

    cfg1 = get_moe_config_for_m(num_tokens, E=E, N=N, H=H, stage="stage1")
    cfg2 = get_moe_config_for_m(num_tokens, E=E, N=N, H=H, stage="stage2")

    _HOST_SPLIT.begin("opaque_alloc")
    intermediate_cache1, intermediate_cache2, intermediate_cache3, out = (
        _ROUTED_WS.acquire(
            num_tokens=num_tokens,
            top_k=top_k,
            N2=N2,
            H=H,
            N=N,
            device=x.device,
            dtype=x.dtype,
        )
    )
    _HOST_SPLIT.end("opaque_alloc")

    _phase_marker("align1")
    _HOST_SPLIT.begin("opaque_align1_wall")
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, int(cfg1["BLOCK_SIZE_M"]), E
    )
    _HOST_SPLIT.end("opaque_align1_wall")
    _retain_capture(sorted_token_ids, expert_ids, num_tokens_post_padded)

    _phase_marker("kernel1")
    _HOST_SPLIT.begin("opaque_kernel1")
    _invoke_kernel(
        x,
        w_gate_up,
        intermediate_cache1,
        topk_w,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        top_k,
        cfg1,
        compute_type,
    )
    _HOST_SPLIT.end("opaque_kernel1")

    _phase_marker("silu")
    _HOST_SPLIT.begin("opaque_silu")
    gate, up = intermediate_cache1.view(-1, N2).chunk(2, dim=-1)
    # #region agent log
    if _DEBUG_STATE["calls"] <= 4 and _under_device_stream_capture():
        _agent_log(
            "H3",
            "fused_moe_runtime.py:opaque_silu",
            "silu_mul_under_capture",
            {
                "call": _DEBUG_STATE["calls"],
                "gate_numel": int(gate.numel()),
                "cache2_data_ptr": int(intermediate_cache2.data_ptr()),
                "gate_data_ptr": int(gate.data_ptr()),
            },
        )
    # #endregion
    # Write silu*up into workspace cache2 (avoid extra temporary + copy_).
    torch.mul(F.silu(gate), up, out=intermediate_cache2)
    _HOST_SPLIT.end("opaque_silu")

    _phase_marker("align2")
    _HOST_SPLIT.begin("opaque_align2_wall")
    # Same BLOCK_SIZE_M → identical align; reuse (common for decode M=1).
    if int(cfg2["BLOCK_SIZE_M"]) == int(cfg1["BLOCK_SIZE_M"]):
        sorted_token_ids2 = sorted_token_ids
        expert_ids2 = expert_ids
        num_tokens_post_padded2 = num_tokens_post_padded
    else:
        sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = moe_align_block_size(
            topk_ids, int(cfg2["BLOCK_SIZE_M"]), E
        )
        _retain_capture(sorted_token_ids2, expert_ids2, num_tokens_post_padded2)
    _HOST_SPLIT.end("opaque_align2_wall")
    _phase_marker("kernel2")
    _HOST_SPLIT.begin("opaque_kernel2")
    _invoke_kernel(
        intermediate_cache2,
        w_down,
        intermediate_cache3,
        topk_w,
        sorted_token_ids2,
        expert_ids2,
        num_tokens_post_padded2,
        True,
        1,
        cfg2,
        compute_type,
    )
    _HOST_SPLIT.end("opaque_kernel2")

    _phase_marker("moe_sum")
    _HOST_SPLIT.begin("opaque_moe_sum")
    moe_sum(intermediate_cache3, out)
    _HOST_SPLIT.end("opaque_moe_sum")
    # Workspace ``out`` is IC-arena-owned under capture; eager path reuses cache.
    return out
