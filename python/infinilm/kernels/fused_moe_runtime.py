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


def _moe_capture_safe_enabled() -> bool:
    """INFINI_MOE_CAPTURE_SAFE=1: aten MoE under stream capture; Triton eager."""
    raw = os.environ.get("INFINI_MOE_CAPTURE_SAFE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _under_device_stream_capture() -> bool:
    try:
        from infinicore.lib import _infinicore as _ic

        return bool(_ic.is_device_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _routed_experts_aten(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Capture-safe routed experts via index_select + bmm (no Triton).

    Same shapes as ``fused_moe_routed``. Used only under ``hcStreamBeginCapture``
    when ``INFINI_MOE_CAPTURE_SAFE=1``; eager serve keeps the Triton path.
    """
    t_tokens, _hidden = x.shape
    del t_tokens, _hidden
    top_k = int(topk_ids.shape[1])
    acc = torch.zeros_like(x)
    for k in range(top_k):
        idx = topk_ids[:, k]
        w_gu = w_gate_up.index_select(0, idx)
        w_d = w_down.index_select(0, idx)
        gu = torch.bmm(w_gu, x.unsqueeze(-1)).squeeze(-1)
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        y = torch.bmm(w_d, h.unsqueeze(-1)).squeeze(-1)
        acc = acc + y * topk_weights[:, k].unsqueeze(-1).to(dtype=x.dtype)
    return acc


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized align (no vLLM ``_custom_ops``, no per-expert Python loop)."""
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = int(flat.numel())
    max_num_tokens_padded = numel + num_experts * (block_size - 1)

    if numel == 0:
        sorted_ids = torch.full(
            (max_num_tokens_padded,), 0, dtype=torch.int32, device=device
        )
        n_m_blocks = 1
        expert_ids_out = torch.full((n_m_blocks,), -1, dtype=torch.int32, device=device)
        num_tokens_post_pad = torch.tensor([0], dtype=torch.int32, device=device)
        return sorted_ids, expert_ids_out, num_tokens_post_pad

    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    idx = torch.arange(numel, device=device, dtype=torch.int64)

    counts = torch.bincount(sorted_experts, minlength=num_experts)
    padded_counts = counts + (block_size - counts % block_size) % block_size
    n_post = int(padded_counts.sum().item())

    padded_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    padded_offsets[1:] = padded_counts.cumsum(0)

    first_pos = torch.full((num_experts,), numel, dtype=torch.int64, device=device)
    first_pos.scatter_reduce_(0, sorted_experts, idx, reduce="amin", include_self=True)

    within_expert = idx - first_pos[sorted_experts]
    out_pos = padded_offsets[sorted_experts] + within_expert

    sorted_ids = torch.full((max_num_tokens_padded,), numel, dtype=torch.int32, device=device)
    sorted_ids.scatter_(0, out_pos, order.to(torch.int32))

    pad_needed = (padded_counts - counts).to(torch.int64)
    total_pad = int(pad_needed.sum().item())
    if total_pad > 0:
        pad_cum = pad_needed.cumsum(0)
        pad_global = torch.arange(total_pad, device=device, dtype=torch.int64)
        pad_expert = torch.arange(num_experts, device=device, dtype=torch.int64).repeat_interleave(
            pad_needed
        )
        pad_cum_start = torch.zeros(num_experts, dtype=torch.int64, device=device)
        pad_cum_start[1:] = pad_cum[:-1]
        pad_within = pad_global - pad_cum_start.repeat_interleave(pad_needed)
        pad_starts = padded_offsets[:-1] + counts
        pad_positions = pad_starts[pad_expert] + pad_within
        sorted_ids[pad_positions] = numel

    n_m_blocks = max((n_post + block_size - 1) // block_size, 1)
    block_counts = (padded_counts // block_size).to(torch.int64)
    expert_ids_full = torch.arange(num_experts, device=device, dtype=torch.int32).repeat_interleave(
        block_counts
    )
    expert_ids_out = torch.full((n_m_blocks,), -1, dtype=torch.int32, device=device)
    if expert_ids_full.numel() > 0:
        expert_ids_out[: expert_ids_full.numel()] = expert_ids_full

    num_tokens_post_pad = torch.tensor([n_post], dtype=torch.int32, device=device)
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


def fused_moe_routed(
    x: torch.Tensor,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Unquantized bf16/fp16 routed experts (Tier-2 entrypoint).

    x: [T, H]; topk_*: [T, K]; w_gate_up [E, 2I, H]; w_down [E, H, I].

    When ``INFINI_MOE_CAPTURE_SAFE=1`` and the stream is under
    ``hcStreamBeginCapture``, uses aten index_select+bmm (capture-safe).
    Otherwise uses Triton cubins (eager / host-break path).
    """
    assert_no_vllm()
    if not x.is_cuda:
        raise RuntimeError("fused_moe_routed requires CUDA/PrivateUse1 tensors")
    assert x.dim() == 2
    assert topk_w.shape == topk_ids.shape
    assert w_gate_up.dim() == 3 and w_down.dim() == 3
    assert x.size(1) == w_gate_up.size(2) == w_down.size(1)

    if _moe_capture_safe_enabled() and _under_device_stream_capture():
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

    cache13 = torch.empty(
        num_tokens * top_k * max(N2, H), device=x.device, dtype=x.dtype
    )
    intermediate_cache1 = cache13[: num_tokens * top_k * N2].view(num_tokens, top_k, N2)
    intermediate_cache3 = cache13[: num_tokens * top_k * H].view(num_tokens, top_k, H)
    intermediate_cache2 = torch.empty(
        (num_tokens * top_k, N), device=x.device, dtype=x.dtype
    )
    out = torch.empty_like(x)

    _phase_marker("align1")
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, int(cfg1["BLOCK_SIZE_M"]), E
    )

    _phase_marker("kernel1")
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

    _phase_marker("silu")
    gate, up = intermediate_cache1.view(-1, N2).chunk(2, dim=-1)
    intermediate_cache2.copy_(F.silu(gate) * up)

    _phase_marker("align2")
    sorted_token_ids2, expert_ids2, num_tokens_post_padded2 = moe_align_block_size(
        topk_ids, int(cfg2["BLOCK_SIZE_M"]), E
    )
    _phase_marker("kernel2")
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

    _phase_marker("moe_sum")
    moe_sum(intermediate_cache3, out)
    return out
