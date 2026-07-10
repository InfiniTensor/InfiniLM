# Copyright (c) 2025, InfiniCore
"""M4 Phase 1: minimal torch mirrors of native piecewise pre/post segments."""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import model_cache_hash
from .env import piecewise_inductor_cache_root

SEGMENT_PRE_ATTN = "pre_attn"
SEGMENT_POST_ATTN_CG = "post_attn_cg"
PIECEWISE_SEGMENT_IDS: Tuple[str, ...] = (SEGMENT_PRE_ATTN, SEGMENT_POST_ATTN_CG)

# Sentinel for layer-agnostic AOT packages (one graph per bucket×tp_rank).
LAYER_AGNOSTIC_IDX = -1
COMPILE_PROFILE_JSONL = "compile_profile.jsonl"


def piecewise_layer_agnostic_enabled() -> bool:
    """When set, compile/register one pre_attn package per bucket×rank (external weights)."""
    return os.environ.get("INFINI_PIECEWISE_LAYER_AGNOSTIC", "1").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def aot_compile_jobs() -> int:
    """TP rank compile jobs are GPU-exclusive; keep at 1 unless sharding across hosts."""
    return max(1, int(os.environ.get("INFINI_AOT_COMPILE_JOBS", "1")))


def piecewise_inductor_shared_cache_dir(
    *,
    cache_root: Optional[str] = None,
    model_path: str,
    tp_rank: int = 0,
) -> str:
    """Per-rank TORCHINDUCTOR cache (avoid rank0 cold-cache kernel drift on HPCC)."""
    root = cache_root or piecewise_inductor_cache_root()
    model_hash = model_cache_hash(model_path)
    return os.path.join(root, model_hash, "inductor_shared", f"rank{int(tp_rank)}")


def _append_compile_profile(
    *,
    cache_root: str,
    model_path: str,
    record: dict,
) -> None:
    profile_path = os.path.join(
        cache_root, model_cache_hash(model_path), COMPILE_PROFILE_JSONL
    )
    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
    with open(profile_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _load_hf_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _global_head_topology(model_path: str, torch_config) -> tuple[int, int, int]:
    """Global head counts from on-disk HF config (torch config may be TP-local)."""
    hf = _load_hf_config(model_path)
    n_heads = int(hf.get("num_attention_heads", torch_config.num_attention_heads))
    n_kv = int(hf.get("num_key_value_heads", torch_config.num_key_value_heads))
    head_dim = int(hf.get("head_dim") or _model_head_dim(torch_config))
    return n_heads, n_kv, head_dim


def _model_head_dim(config) -> int:
    if hasattr(config, "head_dim") and getattr(config, "head_dim", None):
        return int(config.head_dim)
    hidden = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    return hidden // n_heads


def shard_attention_head_dims(
    num_attention_heads: int,
    num_key_value_heads: int,
    tp_size: int,
) -> Tuple[int, int]:
    """Match ``piecewise_prefill_compiler.cpp`` staging head counts per TP rank."""
    tp_size = max(1, int(tp_size))
    n_heads = int(num_attention_heads) // tp_size
    total_kv = int(num_key_value_heads)
    n_kv = 1 if total_kv < tp_size else total_kv // tp_size
    return n_heads, n_kv


def piecewise_inductor_artifact_dir(
    *,
    cache_root: Optional[str] = None,
    model_path: str,
    segment: str,
    layer_idx: int,
    bucket: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    layer_agnostic: bool = False,
) -> str:
    """Directory for one AOT segment artifact.

    Legacy: ``.../tp4/rank0/pre_attn_L0_B512/``
    Layer-agnostic: ``.../tp4/rank0/pre_attn_B512/``
    """
    if segment not in PIECEWISE_SEGMENT_IDS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {PIECEWISE_SEGMENT_IDS}")
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
    root = cache_root or piecewise_inductor_cache_root()
    model_hash = model_cache_hash(model_path)
    if layer_agnostic:
        leaf = f"{segment}_B{int(bucket)}"
    else:
        leaf = f"{segment}_L{int(layer_idx)}_B{int(bucket)}"
    return os.path.join(
        root,
        model_hash,
        f"tp{tp_size}",
        f"rank{tp_rank}",
        leaf,
    )


def _legacy_piecewise_inductor_artifact_dir(
    *,
    cache_root: Optional[str] = None,
    model_path: str,
    segment: str,
    layer_idx: int,
    bucket: int,
) -> str:
    root = cache_root or piecewise_inductor_cache_root()
    model_hash = model_cache_hash(model_path)
    return os.path.join(
        root,
        model_hash,
        f"{segment}_L{int(layer_idx)}_B{int(bucket)}",
    )


def piecewise_inductor_package_path(
    *,
    cache_root: Optional[str] = None,
    model_path: str,
    segment: str,
    layer_idx: int,
    bucket: int,
    tp_size: int = 1,
    tp_rank: int = 0,
    legacy_fallback: bool = True,
    layer_agnostic: Optional[bool] = None,
) -> str:
    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()

    def _pkg(*, layer_agnostic_flag: bool) -> str:
        return os.path.join(
            piecewise_inductor_artifact_dir(
                cache_root=cache_root,
                model_path=model_path,
                segment=segment,
                layer_idx=layer_idx,
                bucket=bucket,
                tp_size=tp_size,
                tp_rank=tp_rank,
                layer_agnostic=layer_agnostic_flag,
            ),
            "segment.pt2",
        )

    if layer_agnostic:
        agnostic = _pkg(layer_agnostic_flag=True)
        if os.path.isfile(agnostic):
            return agnostic
        if legacy_fallback:
            per_layer = _pkg(layer_agnostic_flag=False)
            if os.path.isfile(per_layer):
                return per_layer
        return agnostic

    path = _pkg(layer_agnostic_flag=False)
    if legacy_fallback and tp_size == 1 and tp_rank == 0 and not os.path.isfile(path):
        legacy = os.path.join(
            _legacy_piecewise_inductor_artifact_dir(
                cache_root=cache_root,
                model_path=model_path,
                segment=segment,
                layer_idx=layer_idx,
                bucket=bucket,
            ),
            "segment.pt2",
        )
        if os.path.isfile(legacy):
            return legacy
    return path


def write_piecewise_inductor_metadata(
    artifact_dir: str,
    *,
    model_path: str,
    segment: str,
    layer_idx: int,
    bucket: int,
    extra: Optional[dict] = None,
) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    meta = {
        "model_path": model_path,
        "model_hash": model_cache_hash(model_path),
        "segment": segment,
        "layer_idx": int(layer_idx),
        "bucket": int(bucket),
    }
    if extra:
        meta.update(extra)
    path = os.path.join(artifact_dir, "infinilm_piecewise_segment_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def _rms_norm_last_dim(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Explicit RMSNorm for stable AOTInductor export.

    Use fp32 accumulators so inductor bf16 fusion matches eager on all TP shards
    (rank0 input_layernorm weights previously drifted K by 0.25–2.0).
    """
    xf = x.float()
    wf = weight.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(variance + eps) * wf).to(dtype=x.dtype)


def _fp32_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """QKV matmul in fp32 to avoid inductor bf16 fusion drift on TP shards."""
    return torch.nn.functional.linear(x.float(), weight.float()).to(dtype=x.dtype)


def add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match ``infinicore::op::add_rms_norm`` (residual_out ← a+b, hidden ← RMSNorm)."""
    summed = hidden_states.float() + residual.float()
    normed = _rms_norm_last_dim(summed, weight, eps)
    out_dtype = hidden_states.dtype
    return normed.to(out_dtype), summed.to(out_dtype)


def add_rms_norm_inplace(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    """In-place variant for pinned-buffer replay (Phase 2 hcGraph)."""
    normed, summed = add_rms_norm(hidden_states, residual, weight, eps)
    residual.copy_(summed)
    hidden_states.copy_(normed)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Match ``transformers`` ``rotate_half`` for AOT-safe RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_to_staging(
    staging: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    valid_len: int,
) -> torch.Tensor:
    """Apply RoPE on ``[1, bucket, n_heads, head_dim]`` valid prefix; returns new tensor.

    Functional (no clone+slice mutation) so AOTInductor cannot elide RoPE on rank0.
    """
    valid_len = int(valid_len)
    bucket = int(staging.shape[1])
    if valid_len <= 0:
        return staging
    pre = staging[:, :valid_len]
    view = pre.float().transpose(1, 2).contiguous()
    cos_f = cos[:, :valid_len].float().unsqueeze(1)
    sin_f = sin[:, :valid_len].float().unsqueeze(1)
    rotated = view * cos_f + _rotate_half(view) * sin_f
    roped = rotated.transpose(1, 2).to(dtype=staging.dtype)
    if valid_len >= bucket:
        return roped
    return torch.cat([roped, staging[:, valid_len:]], dim=1)


def _stage_qkv_heads(
    q_heads: torch.Tensor,
    k_heads: torch.Tensor,
    v_heads: torch.Tensor,
    *,
    bucket: int,
    valid_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write Q/K/V head views into staging layout ``[1, bucket, heads, dim]``."""
    valid_len = int(valid_len)
    bucket = int(bucket)
    if valid_len < bucket:
        q_out = torch.zeros_like(q_heads)
        k_out = torch.zeros_like(k_heads)
        v_out = torch.zeros_like(v_heads)
        q_out[:, :valid_len] = q_heads[:, :valid_len]
        k_out[:, :valid_len] = k_heads[:, :valid_len]
        v_out[:, :valid_len] = v_heads[:, :valid_len]
        return q_out, k_out, v_out
    return q_heads, k_heads, v_heads


@dataclass
class PiecewiseSegmentTensors:
    """Staging views written by the pre-attn segment."""

    hidden_states: torch.Tensor
    residual: torch.Tensor
    q_rope: torch.Tensor
    k_rope: torch.Tensor
    v_rope: torch.Tensor


class PiecewisePreAttnSegment(nn.Module):
    """
    Mirror ``TextDecoderLayer::piecewise_pre_attn`` + ``Attention::forward_pre_attn_piecewise``.

    Input RMSNorm (fused add) → QKV → optional Q/K norm → stage Q/K/V → RoPE on staging.
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        rotary_emb: nn.Module,
        *,
        bucket: int,
        valid_seq_len: Optional[int] = None,
        tp_size: int = 1,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
    ):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.bucket = int(bucket)
        self.valid_seq_len = int(valid_seq_len) if valid_seq_len is not None else self.bucket
        self.tp_size = max(1, int(tp_size))
        self.input_layernorm = decoder_layer.input_layernorm
        self.self_attn = decoder_layer.self_attn
        attn_cfg = decoder_layer.self_attn.config
        self.num_attention_heads = int(
            num_attention_heads if num_attention_heads is not None else attn_cfg.num_attention_heads
        )
        self.num_key_value_heads = int(
            num_key_value_heads
            if num_key_value_heads is not None
            else attn_cfg.num_key_value_heads
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from infinilm.torch_llama.rope import segment_position_embeddings

        bucket = int(hidden_states.shape[1])
        valid_len = min(self.valid_seq_len, bucket)
        eps = float(getattr(self.input_layernorm, "variance_epsilon", 1e-6))
        hidden_states, residual = add_rms_norm(
            hidden_states,
            residual,
            self.input_layernorm.weight,
            eps,
        )

        attn = self.self_attn
        q = _fp32_linear(hidden_states, attn.q_proj.weight)
        k = _fp32_linear(hidden_states, attn.k_proj.weight)
        v = _fp32_linear(hidden_states, attn.v_proj.weight)

        n_heads, n_kv = shard_attention_head_dims(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.tp_size,
        )
        head_dim = int(attn.head_dim)

        q_heads = q.view(1, bucket, n_heads, head_dim)
        k_heads = k.view(1, bucket, n_kv, head_dim)
        v_heads = v.view(1, bucket, n_kv, head_dim)

        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        if q_norm is not None:
            q_heads = _rms_norm_last_dim(
                q_heads.reshape(-1, head_dim),
                q_norm.weight,
                float(getattr(q_norm, "variance_epsilon", 1e-6)),
            ).reshape(1, bucket, n_heads, head_dim)
        if k_norm is not None:
            k_heads = _rms_norm_last_dim(
                k_heads.reshape(-1, head_dim),
                k_norm.weight,
                float(getattr(k_norm, "variance_epsilon", 1e-6)),
            ).reshape(1, bucket, n_kv, head_dim)

        q_staging, k_staging, v_staging = _stage_qkv_heads(
            q_heads,
            k_heads,
            v_heads,
            bucket=bucket,
            valid_len=valid_len,
        )

        cos, sin = segment_position_embeddings(
            self.rotary_emb,
            hidden_states,
            position_ids,
            valid_len=valid_len,
        )
        q_staging = _apply_rope_to_staging(
            q_staging, cos=cos, sin=sin, valid_len=valid_len
        )
        k_staging = _apply_rope_to_staging(
            k_staging, cos=cos, sin=sin, valid_len=valid_len
        )

        return hidden_states, residual, q_staging, k_staging, v_staging


class PiecewisePreAttnSegmentFunctional(nn.Module):
    """Layer-agnostic pre-attn segment: weights passed as forward inputs (Phase 2)."""

    def __init__(
        self,
        rotary_emb: nn.Module,
        *,
        bucket: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        valid_seq_len: Optional[int] = None,
        tp_size: int = 1,
        has_qk_norm: bool = True,
    ):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.bucket = int(bucket)
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.rms_norm_eps = float(rms_norm_eps)
        self.valid_seq_len = int(valid_seq_len) if valid_seq_len is not None else self.bucket
        self.tp_size = max(1, int(tp_size))
        self.has_qk_norm = bool(has_qk_norm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        position_ids: torch.Tensor,
        ln_weight: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from infinilm.torch_llama.rope import segment_position_embeddings

        bucket = int(hidden_states.shape[1])
        valid_len = min(self.valid_seq_len, bucket)
        n_heads, n_kv = shard_attention_head_dims(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.tp_size,
        )

        hidden_states, residual = add_rms_norm(
            hidden_states,
            residual,
            ln_weight,
            self.rms_norm_eps,
        )

        q = _fp32_linear(hidden_states, q_weight)
        k = _fp32_linear(hidden_states, k_weight)
        v = _fp32_linear(hidden_states, v_weight)

        q_heads = q.view(1, bucket, n_heads, self.head_dim)
        k_heads = k.view(1, bucket, n_kv, self.head_dim)
        v_heads = v.view(1, bucket, n_kv, self.head_dim)

        if self.has_qk_norm and q_norm_weight.numel() > 0:
            q_heads = _rms_norm_last_dim(
                q_heads.reshape(-1, self.head_dim),
                q_norm_weight,
                self.rms_norm_eps,
            ).reshape(1, bucket, n_heads, self.head_dim)
        if self.has_qk_norm and k_norm_weight.numel() > 0:
            k_heads = _rms_norm_last_dim(
                k_heads.reshape(-1, self.head_dim),
                k_norm_weight,
                self.rms_norm_eps,
            ).reshape(1, bucket, n_kv, self.head_dim)

        q_staging, k_staging, v_staging = _stage_qkv_heads(
            q_heads,
            k_heads,
            v_heads,
            bucket=bucket,
            valid_len=valid_len,
        )

        cos, sin = segment_position_embeddings(
            self.rotary_emb,
            hidden_states,
            position_ids,
            valid_len=valid_len,
        )
        q_staging = _apply_rope_to_staging(
            q_staging, cos=cos, sin=sin, valid_len=valid_len
        )
        k_staging = _apply_rope_to_staging(
            k_staging, cos=cos, sin=sin, valid_len=valid_len
        )

        return hidden_states, residual, q_staging, k_staging, v_staging


def _extract_pre_attn_weights(decoder_layer: nn.Module, device: torch.device, dtype: torch.dtype):
    """Weight tensors for ``PiecewisePreAttnSegmentFunctional`` example inputs."""
    attn = decoder_layer.self_attn
    ln_weight = decoder_layer.input_layernorm.weight.detach().to(device=device, dtype=dtype)
    q_weight = attn.q_proj.weight.detach().to(device=device, dtype=dtype)
    k_weight = attn.k_proj.weight.detach().to(device=device, dtype=dtype)
    v_weight = attn.v_proj.weight.detach().to(device=device, dtype=dtype)
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    empty = torch.empty(0, device=device, dtype=dtype)
    q_norm_weight = (
        q_norm.weight.detach().to(device=device, dtype=dtype) if q_norm is not None else empty
    )
    k_norm_weight = (
        k_norm.weight.detach().to(device=device, dtype=dtype) if k_norm is not None else empty
    )
    has_qk_norm = q_norm is not None and k_norm is not None
    return ln_weight, q_weight, k_weight, v_weight, q_norm_weight, k_norm_weight, has_qk_norm


class PiecewisePostAttnCgSegment(nn.Module):
    """Mirror ``piecewise_post_attn_cg`` (O-proj + post RMSNorm + MLP). Deferred past Phase 1 gate."""

    def __init__(self, decoder_layer: nn.Module, *, bucket: int, valid_seq_len: Optional[int] = None):
        super().__init__()
        raise NotImplementedError(
            "post_attn_cg segment AOT is Phase 1+; implement after pre_attn gate passes"
        )


def build_piecewise_segment(
    torch_model,
    *,
    segment: str,
    layer_idx: int,
    bucket: int,
    valid_seq_len: Optional[int] = None,
    tp_size: int = 1,
    layer_agnostic: bool = False,
    model_path: Optional[str] = None,
) -> nn.Module:
    """Construct a segment module from a weight-bound ``TorchLlamaPrefillModel``."""
    inner = torch_model.inner.model
    decoder_layer = inner.layers[int(layer_idx)]
    config = torch_model.config
    if model_path:
        global_n_heads, global_n_kv, head_dim = _global_head_topology(model_path, config)
    else:
        global_n_heads = int(config.num_attention_heads)
        global_n_kv = int(config.num_key_value_heads)
        head_dim = _model_head_dim(config)
    if segment == SEGMENT_PRE_ATTN:
        if layer_agnostic:
            has_qk_norm = (
                getattr(decoder_layer.self_attn, "q_norm", None) is not None
                and getattr(decoder_layer.self_attn, "k_norm", None) is not None
            )
            return PiecewisePreAttnSegmentFunctional(
                inner.rotary_emb,
                bucket=bucket,
                hidden_size=int(config.hidden_size),
                num_attention_heads=global_n_heads,
                num_key_value_heads=global_n_kv,
                head_dim=head_dim,
                rms_norm_eps=float(getattr(decoder_layer.input_layernorm, "variance_epsilon", 1e-6)),
                valid_seq_len=valid_seq_len,
                tp_size=tp_size,
                has_qk_norm=has_qk_norm,
            )
        return PiecewisePreAttnSegment(
            decoder_layer,
            inner.rotary_emb,
            bucket=bucket,
            valid_seq_len=valid_seq_len,
            tp_size=tp_size,
            num_attention_heads=global_n_heads,
            num_key_value_heads=global_n_kv,
        )
    if segment == SEGMENT_POST_ATTN_CG:
        return PiecewisePostAttnCgSegment(
            decoder_layer,
            bucket=bucket,
            valid_seq_len=valid_seq_len,
        )
    raise ValueError(f"unknown segment {segment!r}")


def make_segment_example_inputs(
    *,
    bucket: int,
    hidden_size: int,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    valid_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """Random example tensors for ``torch.export`` / AOTInductor."""
    valid = int(valid_seq_len) if valid_seq_len is not None else int(bucket)
    hidden = torch.randn(1, bucket, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(1, bucket, hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(valid, device=device, dtype=torch.long).unsqueeze(0)
    if valid < bucket:
        pad = torch.zeros(1, bucket - valid, device=device, dtype=torch.long)
        position_ids = torch.cat([position_ids, pad], dim=1)
    return hidden, residual, position_ids


def load_torch_model_with_cpp_weights(
    model_path: str,
    device: torch.device,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_device_ids: Optional[Sequence[int]] = None,
):
    """InferEngine weight blobs → ``TorchLlamaPrefillModel`` (M3 pattern)."""
    import infinicore

    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file
    from infinilm.torch_llama.model import load_torch_llama

    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    if tp_device_ids is None:
        tp_device_ids = list(range(tp_size))
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")

    if tp_size > 1:
        dist_cfg = DistConfig(tp_device_ids=list(tp_device_ids))
        engine_device = infinicore.device("cuda", int(tp_device_ids[0]))
    else:
        dist_cfg = DistConfig(1)
        engine_device = infinicore.device(
            "cuda", device.index if device.index is not None else 0
        )

    engine = InferEngine(
        model_path,
        device=engine_device,
        distributed_config=dist_cfg,
        enable_graph_compiling=False,
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)
    cpp_state_dict = engine._cpp_state_dict_for_compile(tp_rank)
    del engine
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if device.type == "cuda":
        torch_device = torch.device("cuda", int(tp_device_ids[tp_rank]))
    else:
        torch_device = device

    return load_torch_llama(
        model_path,
        device=torch_device,
        splitting_flash_boundary=False,
        cpp_state_dict=cpp_state_dict,
        tp_size=tp_size,
    )


def segment_output_fingerprint(
    q_staging: torch.Tensor,
    k_staging: torch.Tensor,
    v_staging: torch.Tensor,
    *,
    valid_len: int,
) -> torch.Tensor:
    """Summary vector for token_match gate (last valid token, flattened Q/K/V)."""
    valid_len = int(valid_len)
    parts = [
        q_staging[0, valid_len - 1].reshape(-1),
        k_staging[0, valid_len - 1].reshape(-1),
        v_staging[0, valid_len - 1].reshape(-1),
    ]
    return torch.cat(parts, dim=0).float()


def aot_compile_piecewise_segment(
    *,
    model_path: str,
    segment: str,
    layer_idx: int,
    bucket: int,
    device: torch.device,
    cache_root: Optional[str] = None,
    valid_seq_len: Optional[int] = None,
    require_aot: bool = False,
    torch_model=None,
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_device_ids: Optional[Sequence[int]] = None,
    layer_agnostic: Optional[bool] = None,
    profile_compile: bool = True,
) -> dict:
    """``torch.export`` + AOTInductor package for one piecewise segment."""
    from .aot_hpcc_patch import (
        build_aot_inductor_configs,
        hpcc_aot_compile_profile,
        is_hpcc_aot_environment,
        patch_wrapper_sources,
    )
    from .env import piecewise_inductor_require_aot

    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()

    t_total0 = time.perf_counter()
    profile: dict = {
        "segment": segment,
        "layer_idx": int(layer_idx),
        "bucket": int(bucket),
        "tp_size": int(tp_size),
        "tp_rank": int(tp_rank),
        "layer_agnostic": bool(layer_agnostic),
    }

    require_aot = require_aot or piecewise_inductor_require_aot()
    valid = int(valid_seq_len) if valid_seq_len is not None else int(bucket)
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    root = cache_root or piecewise_inductor_cache_root()

    t0 = time.perf_counter()
    if torch_model is None:
        torch_model = load_torch_model_with_cpp_weights(
            model_path,
            device,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_device_ids=tp_device_ids,
        )
    profile["weight_load_ms"] = (time.perf_counter() - t0) * 1000.0

    config = torch_model.config
    global_n_heads, global_n_kv, head_dim = _global_head_topology(model_path, config)
    n_heads, n_kv = shard_attention_head_dims(global_n_heads, global_n_kv, tp_size)
    hidden_size = int(config.hidden_size)
    dtype = next(torch_model.inner.parameters()).dtype

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    t0 = time.perf_counter()
    ref_layer_idx = 0 if layer_agnostic else int(layer_idx)
    segment_module = build_piecewise_segment(
        torch_model,
        segment=segment,
        layer_idx=ref_layer_idx,
        bucket=bucket,
        valid_seq_len=valid,
        tp_size=tp_size,
        layer_agnostic=layer_agnostic,
        model_path=model_path,
    ).eval()
    segment_module.requires_grad_(False)
    profile["build_segment_ms"] = (time.perf_counter() - t0) * 1000.0

    example_inputs = make_segment_example_inputs(
        bucket=bucket,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_kv=n_kv,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        valid_seq_len=valid,
    )
    if layer_agnostic and segment == SEGMENT_PRE_ATTN:
        inner = torch_model.inner.model
        weight_inputs = _extract_pre_attn_weights(
            inner.layers[ref_layer_idx], device, dtype
        )
        example_inputs = example_inputs + tuple(
            t.detach().requires_grad_(False) for t in weight_inputs[:6]
        )
    example_inputs = tuple(t.detach().requires_grad_(False) for t in example_inputs)

    meta_layer_idx = LAYER_AGNOSTIC_IDX if layer_agnostic else int(layer_idx)
    artifact_dir = os.path.abspath(
        piecewise_inductor_artifact_dir(
            cache_root=root,
            model_path=model_path,
            segment=segment,
            layer_idx=meta_layer_idx,
            bucket=bucket,
            tp_size=tp_size,
            tp_rank=tp_rank,
            layer_agnostic=layer_agnostic,
        )
    )
    os.makedirs(artifact_dir, exist_ok=True)
    package_path = os.path.abspath(
        piecewise_inductor_package_path(
            cache_root=root,
            model_path=model_path,
            segment=segment,
            layer_idx=meta_layer_idx,
            bucket=bucket,
            tp_size=tp_size,
            tp_rank=tp_rank,
            legacy_fallback=False,
            layer_agnostic=layer_agnostic,
        )
    )

    _meta_common = {
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "n_heads_local": n_heads,
        "n_kv_heads_local": n_kv,
        "layer_agnostic": layer_agnostic,
    }

    inductor_cache = piecewise_inductor_shared_cache_dir(
        cache_root=root, model_path=model_path, tp_rank=tp_rank
    )
    os.makedirs(inductor_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath(inductor_cache)

    inductor_configs = None
    if is_hpcc_aot_environment():
        inductor_configs = build_aot_inductor_configs(artifact_dir=artifact_dir)

    t0 = time.perf_counter()
    with torch.no_grad():
        exported = torch.export.export(
            segment_module,
            example_inputs,
            strict=False,
        )
    profile["export_ms"] = (time.perf_counter() - t0) * 1000.0

    from torch._inductor import aoti_compile_and_package

    def _run_aot_package():
        ctx = (
            hpcc_aot_compile_profile(artifact_dir=artifact_dir)
            if is_hpcc_aot_environment()
            else contextlib.nullcontext({})
        )
        with ctx:
            if inductor_configs:
                return aoti_compile_and_package(
                    exported,
                    package_path=package_path,
                    inductor_configs=inductor_configs,
                )
            return aoti_compile_and_package(exported, package_path=package_path)

    t0 = time.perf_counter()
    try:
        compiled_path = _run_aot_package()
        backend = "aot_inductor"
    except Exception as exc:  # noqa: BLE001
        profile["aot_package_ms"] = (time.perf_counter() - t0) * 1000.0
        # P1 retry: patch any emitted wrapper sources and retry once.
        if is_hpcc_aot_environment():
            patched = patch_wrapper_sources(inductor_cache)
            if patched > 0:
                try:
                    t_retry = time.perf_counter()
                    compiled_path = _run_aot_package()
                    backend = "aot_inductor"
                    profile["aot_package_ms"] = (time.perf_counter() - t_retry) * 1000.0
                except Exception:
                    pass
                else:
                    meta_path = write_piecewise_inductor_metadata(
                        artifact_dir,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=meta_layer_idx,
                        bucket=bucket,
                        extra={
                            **_meta_common,
                            "package_path": compiled_path,
                            "valid_seq_len": valid,
                            "hidden_size": hidden_size,
                            "num_attention_heads": n_heads,
                            "num_key_value_heads": n_kv,
                            "head_dim": head_dim,
                            "dtype": str(dtype),
                            "backend": backend,
                            "hpcc_wrapper_patch": patched,
                        },
                    )
                    profile["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
                    profile["backend"] = backend
                    if profile_compile:
                        _append_compile_profile(
                            cache_root=root, model_path=model_path, record=profile
                        )
                    return {
                        "artifact_dir": artifact_dir,
                        "package_path": compiled_path,
                        "meta_path": meta_path,
                        "backend": backend,
                        "profile": profile,
                    }

        if require_aot:
            profile["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
            profile["error"] = str(exc)
            if profile_compile:
                _append_compile_profile(cache_root=root, model_path=model_path, record=profile)
            raise RuntimeError(
                f"AOTInductor packaging required but failed: {exc}"
            ) from exc

        meta_path = write_piecewise_inductor_metadata(
            artifact_dir,
            model_path=model_path,
            segment=segment,
            layer_idx=meta_layer_idx,
            bucket=bucket,
            extra={
                **_meta_common,
                "package_path": "",
                "aot_error": str(exc),
                "valid_seq_len": valid,
                "hidden_size": hidden_size,
                "num_attention_heads": n_heads,
                "num_key_value_heads": n_kv,
                "head_dim": head_dim,
                "dtype": str(dtype),
                "backend": "torch_compile_fallback",
            },
        )
        compiled_fn = torch_compile_piecewise_segment(segment_module, example_inputs)
        fallback_path = os.path.join(artifact_dir, "segment_torch_compile.pt")
        torch.save(
            {
                "state_dict": segment_module.state_dict(),
                "bucket": bucket,
                "valid_seq_len": valid,
                "segment": segment,
                "layer_idx": meta_layer_idx,
            },
            fallback_path,
        )
        profile["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
        profile["backend"] = "torch_compile"
        if profile_compile:
            _append_compile_profile(cache_root=root, model_path=model_path, record=profile)
        return {
            "artifact_dir": artifact_dir,
            "package_path": "",
            "meta_path": meta_path,
            "backend": "torch_compile",
            "compiled_fn": compiled_fn,
            "fallback_path": fallback_path,
            "aot_error": str(exc),
            "profile": profile,
        }

    profile["aot_package_ms"] = (time.perf_counter() - t0) * 1000.0
    meta_path = write_piecewise_inductor_metadata(
        artifact_dir,
        model_path=model_path,
        segment=segment,
        layer_idx=meta_layer_idx,
        bucket=bucket,
        extra={
            **_meta_common,
            "package_path": compiled_path,
            "valid_seq_len": valid,
            "hidden_size": hidden_size,
            "num_attention_heads": n_heads,
            "num_key_value_heads": n_kv,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "backend": backend,
        },
    )
    profile["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
    profile["backend"] = backend
    if profile_compile:
        _append_compile_profile(cache_root=root, model_path=model_path, record=profile)
    return {
        "artifact_dir": artifact_dir,
        "package_path": compiled_path,
        "meta_path": meta_path,
        "backend": backend,
        "profile": profile,
    }


def aot_compile_piecewise_segments_batch(
    *,
    model_path: str,
    segment: str,
    layer_indices: Sequence[int],
    bucket: int,
    device: torch.device,
    cache_root: Optional[str] = None,
    valid_seq_len: Optional[int] = None,
    require_aot: bool = False,
    skip_existing: bool = True,
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_device_ids: Optional[Sequence[int]] = None,
    layer_agnostic: Optional[bool] = None,
) -> list:
    """Compile multiple layers reusing one weight load."""
    from .env import piecewise_inductor_require_aot

    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()

    require_aot = require_aot or piecewise_inductor_require_aot()
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    root = cache_root or piecewise_inductor_cache_root()

    if layer_agnostic:
        meta_layer_idx = LAYER_AGNOSTIC_IDX
        if skip_existing:
            package_path = piecewise_inductor_package_path(
                cache_root=root,
                model_path=model_path,
                segment=segment,
                layer_idx=meta_layer_idx,
                bucket=bucket,
                tp_size=tp_size,
                tp_rank=tp_rank,
                layer_agnostic=True,
            )
            if os.path.isfile(package_path):
                print(
                    f"[aot_batch] skip layer-agnostic B{bucket} tp{tp_size}/rank{tp_rank} exists",
                    flush=True,
                )
                return []
        layer_indices = [int(layer_indices[0]) if layer_indices else 0]

    to_compile: list[int] = []
    for layer_idx in layer_indices:
        if skip_existing and not layer_agnostic:
            package_path = piecewise_inductor_package_path(
                cache_root=root,
                model_path=model_path,
                segment=segment,
                layer_idx=layer_idx,
                bucket=bucket,
                tp_size=tp_size,
                tp_rank=tp_rank,
                layer_agnostic=False,
            )
            if os.path.isfile(package_path):
                print(
                    f"[aot_batch] skip L{layer_idx} B{bucket} tp{tp_size}/rank{tp_rank} exists",
                    flush=True,
                )
                continue
        to_compile.append(int(layer_idx))

    if not to_compile:
        return []

    print(
        f"[aot_batch] loading weights once for {len(to_compile)} layer(s) "
        f"B{bucket} tp{tp_size}/rank{tp_rank} layer_agnostic={layer_agnostic}",
        flush=True,
    )
    t_load0 = time.perf_counter()
    torch_model = load_torch_model_with_cpp_weights(
        model_path,
        device,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_device_ids=tp_device_ids,
    )
    print(
        f"[aot_batch] weight_load_ms={(time.perf_counter() - t_load0) * 1000.0:.1f}",
        flush=True,
    )
    results = []
    try:
        for layer_idx in to_compile:
            print(
                f"[aot_batch] L{layer_idx} B{bucket} tp{tp_size}/rank{tp_rank} "
                f"start {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
                flush=True,
            )
            t0 = time.perf_counter()
            summary = aot_compile_piecewise_segment(
                model_path=model_path,
                segment=segment,
                layer_idx=layer_idx,
                bucket=bucket,
                device=device,
                cache_root=root,
                valid_seq_len=valid_seq_len,
                require_aot=require_aot,
                torch_model=torch_model,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_device_ids=tp_device_ids,
                layer_agnostic=layer_agnostic,
            )
            summary["total_ms"] = (time.perf_counter() - t0) * 1000.0
            prof = summary.get("profile") or {}
            print(
                f"[aot_batch] L{layer_idx} B{bucket} PASS backend={summary.get('backend', '')} "
                f"total_ms={summary['total_ms']:.1f} "
                f"export_ms={prof.get('export_ms', 0):.1f} "
                f"aot_ms={prof.get('aot_package_ms', 0):.1f}",
                flush=True,
            )
            results.append(summary)
            if layer_agnostic:
                break
    finally:
        del torch_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def aot_compile_piecewise_segments_multi_bucket(
    *,
    model_path: str,
    segment: str,
    buckets: Sequence[int],
    device: torch.device,
    cache_root: Optional[str] = None,
    layer_indices: Optional[Sequence[int]] = None,
    valid_seq_len: Optional[int] = None,
    require_aot: bool = False,
    skip_existing: bool = True,
    tp_size: int = 1,
    tp_rank: int = 0,
    tp_device_ids: Optional[Sequence[int]] = None,
    layer_agnostic: Optional[bool] = None,
) -> list:
    """One weight load per rank; compile all buckets (and layers unless layer-agnostic)."""
    import json as _json

    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()

    config_path = os.path.join(model_path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        num_layers = int(_json.load(f)["num_hidden_layers"])
    if layer_indices is None:
        layer_indices = list(range(num_layers))

    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    root = cache_root or piecewise_inductor_cache_root()
    bucket_list = [int(b) for b in buckets]

    print(
        f"[aot_multi] segment={segment} buckets={bucket_list} tp{tp_size}/rank{tp_rank} "
        f"layer_agnostic={layer_agnostic} layers={len(layer_indices)}",
        flush=True,
    )
    t_load0 = time.perf_counter()
    torch_model = load_torch_model_with_cpp_weights(
        model_path,
        device,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_device_ids=tp_device_ids,
    )
    print(
        f"[aot_multi] weight_load_ms={(time.perf_counter() - t_load0) * 1000.0:.1f}",
        flush=True,
    )

    results: list = []
    try:
        for bucket in bucket_list:
            valid = int(valid_seq_len) if valid_seq_len is not None else int(bucket)
            indices = layer_indices
            if layer_agnostic:
                indices = [0]
                meta_layer_idx = LAYER_AGNOSTIC_IDX
                if skip_existing:
                    pkg = piecewise_inductor_package_path(
                        cache_root=root,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=meta_layer_idx,
                        bucket=bucket,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        layer_agnostic=True,
                    )
                    if os.path.isfile(pkg):
                        print(
                            f"[aot_multi] skip layer-agnostic B{bucket} tp{tp_size}/rank{tp_rank}",
                            flush=True,
                        )
                        continue
            else:
                indices = [
                    idx
                    for idx in layer_indices
                    if not (
                        skip_existing
                        and os.path.isfile(
                            piecewise_inductor_package_path(
                                cache_root=root,
                                model_path=model_path,
                                segment=segment,
                                layer_idx=idx,
                                bucket=bucket,
                                tp_size=tp_size,
                                tp_rank=tp_rank,
                                layer_agnostic=False,
                            )
                        )
                    )
                ]
                if not indices:
                    print(
                        f"[aot_multi] skip B{bucket} all layers exist tp{tp_size}/rank{tp_rank}",
                        flush=True,
                    )
                    continue

            for layer_idx in indices:
                print(
                    f"[aot_multi] B{bucket} L{layer_idx} tp{tp_size}/rank{tp_rank} start",
                    flush=True,
                )
                t0 = time.perf_counter()
                summary = aot_compile_piecewise_segment(
                    model_path=model_path,
                    segment=segment,
                    layer_idx=int(layer_idx),
                    bucket=bucket,
                    device=device,
                    cache_root=root,
                    valid_seq_len=valid,
                    require_aot=require_aot,
                    torch_model=torch_model,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    tp_device_ids=tp_device_ids,
                    layer_agnostic=layer_agnostic,
                )
                summary["total_ms"] = (time.perf_counter() - t0) * 1000.0
                results.append(summary)
                if layer_agnostic:
                    break
    finally:
        del torch_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def expected_piecewise_package_count(
    *,
    num_layers: int,
    buckets: Sequence[int],
    tp_size: int = 1,
    layer_agnostic: Optional[bool] = None,
) -> int:
    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()
    tp_size = max(1, int(tp_size))
    if layer_agnostic:
        return len(buckets) * tp_size
    return num_layers * len(buckets) * tp_size


def sub512_farm_buckets() -> Tuple[int, ...]:
    """Sub-512 vLLM tail ladder (1,2,4,...,512) for layer-agnostic AOT farm."""
    from .env import vllm_piecewise_capture_sizes

    return vllm_piecewise_capture_sizes(512)


def vllm_aligned_farm_buckets(max_seq_len: Optional[int] = None) -> Tuple[int, ...]:
    """Full vLLM-aligned capture ladder: sub-512 + power buckets through 8192."""
    from .env import compile_max_seq_len, native_piecewise_capture_buckets_vllm, prefill_chunk_size

    max_seq = int(max_seq_len) if max_seq_len is not None else compile_max_seq_len()
    return native_piecewise_capture_buckets_vllm(max_seq, prefill_chunk_size(default=512))


def torch_compile_piecewise_segment(
    segment_module: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    *,
    compile_mode: str = "default",
) -> nn.Module:
    """Runtime ``torch.compile`` wrapper (Phase 1 parity when AOT packaging fails)."""
    segment_module = segment_module.eval().requires_grad_(False)
    inputs = tuple(t.detach().requires_grad_(False) for t in example_inputs)
    with torch.no_grad():
        compiled = torch.compile(segment_module, mode=compile_mode, fullgraph=False)
        # Warmup once so parity timing excludes compile latency.
        _ = compiled(*inputs)
        if inputs[0].is_cuda:
            torch.cuda.synchronize()
    return compiled
