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
) -> str:
    """Directory for one AOT segment artifact: ``.../tp2/rank0/pre_attn_L0_B512/``."""
    if segment not in PIECEWISE_SEGMENT_IDS:
        raise ValueError(f"unknown segment {segment!r}; expected one of {PIECEWISE_SEGMENT_IDS}")
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
    root = cache_root or piecewise_inductor_cache_root()
    model_hash = model_cache_hash(model_path)
    return os.path.join(
        root,
        model_hash,
        f"tp{tp_size}",
        f"rank{tp_rank}",
        f"{segment}_L{int(layer_idx)}_B{int(bucket)}",
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
) -> str:
    path = os.path.join(
        piecewise_inductor_artifact_dir(
            cache_root=cache_root,
            model_path=model_path,
            segment=segment,
            layer_idx=layer_idx,
            bucket=bucket,
            tp_size=tp_size,
            tp_rank=tp_rank,
        ),
        "segment.pt2",
    )
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
    if hasattr(torch.nn.functional, "rms_norm"):
        return torch.nn.functional.rms_norm(
            x,
            (x.shape[-1],),
            weight,
            eps,
        )
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def add_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match ``infinicore::op::add_rms_norm`` (residual_out ← a+b, hidden ← RMSNorm)."""
    summed = hidden_states + residual
    normed = _rms_norm_last_dim(summed, weight, eps)
    return normed, summed


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


def _apply_rope_to_staging(
    staging: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    valid_len: int,
) -> torch.Tensor:
    """Apply RoPE on ``[1, bucket, n_heads, head_dim]`` valid prefix; returns new tensor."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    valid_len = int(valid_len)
    out = staging.clone()
    if valid_len <= 0:
        return out
    view = staging[:, :valid_len].transpose(1, 2).contiguous()
    rotated, _ = apply_rotary_pos_emb(
        view,
        view,
        cos[:, :valid_len],
        sin[:, :valid_len],
        unsqueeze_dim=1,
    )
    out[:, :valid_len] = rotated.transpose(1, 2)
    return out


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
    ):
        super().__init__()
        self.rotary_emb = rotary_emb
        self.bucket = int(bucket)
        self.valid_seq_len = int(valid_seq_len) if valid_seq_len is not None else self.bucket
        self.tp_size = max(1, int(tp_size))
        self.input_layernorm = decoder_layer.input_layernorm
        self.self_attn = decoder_layer.self_attn

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
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        n_heads, n_kv = shard_attention_head_dims(
            int(attn.config.num_attention_heads),
            int(attn.config.num_key_value_heads),
            self.tp_size,
        )
        head_dim = int(attn.head_dim)

        q_heads = q.view(1, bucket, n_heads, head_dim)
        k_heads = k.view(1, bucket, n_kv, head_dim)
        v_heads = v.view(1, bucket, n_kv, head_dim)

        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)
        if q_norm is not None:
            q_heads = q_norm(q_heads.reshape(-1, head_dim)).reshape(1, bucket, n_heads, head_dim)
        if k_norm is not None:
            k_heads = k_norm(k_heads.reshape(-1, head_dim)).reshape(1, bucket, n_kv, head_dim)

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
) -> nn.Module:
    """Construct a segment module from a weight-bound ``TorchLlamaPrefillModel``."""
    inner = torch_model.inner.model
    decoder_layer = inner.layers[int(layer_idx)]
    if segment == SEGMENT_PRE_ATTN:
        return PiecewisePreAttnSegment(
            decoder_layer,
            inner.rotary_emb,
            bucket=bucket,
            valid_seq_len=valid_seq_len,
            tp_size=tp_size,
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
) -> dict:
    """``torch.export`` + AOTInductor package for one piecewise segment."""
    from .aot_hpcc_patch import (
        build_aot_inductor_configs,
        hpcc_aot_compile_profile,
        is_hpcc_aot_environment,
        patch_wrapper_sources,
    )
    from .env import piecewise_inductor_require_aot

    require_aot = require_aot or piecewise_inductor_require_aot()
    valid = int(valid_seq_len) if valid_seq_len is not None else int(bucket)
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    root = cache_root or piecewise_inductor_cache_root()
    if torch_model is None:
        torch_model = load_torch_model_with_cpp_weights(
            model_path,
            device,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_device_ids=tp_device_ids,
        )
    config = torch_model.config
    hidden_size = int(config.hidden_size)
    n_heads, n_kv = shard_attention_head_dims(
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        tp_size,
    )
    head_dim = hidden_size // int(config.num_attention_heads)
    dtype = next(torch_model.inner.parameters()).dtype

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    segment_module = build_piecewise_segment(
        torch_model,
        segment=segment,
        layer_idx=layer_idx,
        bucket=bucket,
        valid_seq_len=valid,
        tp_size=1,
    ).eval()
    segment_module.requires_grad_(False)

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
    example_inputs = tuple(t.detach().requires_grad_(False) for t in example_inputs)

    artifact_dir = os.path.abspath(
        piecewise_inductor_artifact_dir(
            cache_root=root,
            model_path=model_path,
            segment=segment,
            layer_idx=layer_idx,
            bucket=bucket,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
    )
    os.makedirs(artifact_dir, exist_ok=True)
    package_path = os.path.abspath(
        piecewise_inductor_package_path(
            cache_root=root,
            model_path=model_path,
            segment=segment,
            layer_idx=layer_idx,
            bucket=bucket,
            tp_size=tp_size,
            tp_rank=tp_rank,
            legacy_fallback=False,
        )
    )

    _meta_common = {
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "n_heads_local": n_heads,
        "n_kv_heads_local": n_kv,
    }

    inductor_cache = os.path.join(artifact_dir, "inductor")
    os.makedirs(inductor_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath(inductor_cache)

    inductor_configs = None
    if is_hpcc_aot_environment():
        inductor_configs = build_aot_inductor_configs(artifact_dir=artifact_dir)

    with torch.no_grad():
        exported = torch.export.export(
            segment_module,
            example_inputs,
            strict=False,
        )

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

    try:
        compiled_path = _run_aot_package()
        backend = "aot_inductor"
    except Exception as exc:  # noqa: BLE001
        # P1 retry: patch any emitted wrapper sources and retry once.
        if is_hpcc_aot_environment():
            patched = patch_wrapper_sources(inductor_cache)
            if patched > 0:
                try:
                    compiled_path = _run_aot_package()
                    backend = "aot_inductor"
                except Exception:
                    pass
                else:
                    meta_path = write_piecewise_inductor_metadata(
                        artifact_dir,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=layer_idx,
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
                    return {
                        "artifact_dir": artifact_dir,
                        "package_path": compiled_path,
                        "meta_path": meta_path,
                        "backend": backend,
                    }

        if require_aot:
            raise RuntimeError(
                f"AOTInductor packaging required but failed: {exc}"
            ) from exc

        meta_path = write_piecewise_inductor_metadata(
            artifact_dir,
            model_path=model_path,
            segment=segment,
            layer_idx=layer_idx,
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
                "layer_idx": layer_idx,
            },
            fallback_path,
        )
        return {
            "artifact_dir": artifact_dir,
            "package_path": "",
            "meta_path": meta_path,
            "backend": "torch_compile",
            "compiled_fn": compiled_fn,
            "fallback_path": fallback_path,
            "aot_error": str(exc),
        }

    meta_path = write_piecewise_inductor_metadata(
        artifact_dir,
        model_path=model_path,
        segment=segment,
        layer_idx=layer_idx,
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
    return {
        "artifact_dir": artifact_dir,
        "package_path": compiled_path,
        "meta_path": meta_path,
        "backend": backend,
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
) -> list:
    """Compile multiple layers reusing one weight load."""
    from .env import piecewise_inductor_require_aot

    require_aot = require_aot or piecewise_inductor_require_aot()
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)
    root = cache_root or piecewise_inductor_cache_root()
    to_compile: list[int] = []
    for layer_idx in layer_indices:
        if skip_existing:
            package_path = piecewise_inductor_package_path(
                cache_root=root,
                model_path=model_path,
                segment=segment,
                layer_idx=layer_idx,
                bucket=bucket,
                tp_size=tp_size,
                tp_rank=tp_rank,
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
        f"B{bucket} tp{tp_size}/rank{tp_rank}",
        flush=True,
    )
    torch_model = load_torch_model_with_cpp_weights(
        model_path,
        device,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_device_ids=tp_device_ids,
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
            )
            summary["total_ms"] = (time.perf_counter() - t0) * 1000.0
            print(
                f"[aot_batch] L{layer_idx} B{bucket} PASS backend={summary.get('backend', '')} "
                f"total_ms={summary['total_ms']:.1f}",
                flush=True,
            )
            results.append(summary)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del torch_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


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
