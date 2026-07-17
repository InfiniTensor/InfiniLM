# Copyright (c) 2025, InfiniCore
"""Layer-agnostic MiniCPM5 MoE segment for AOTInductor (Track B; no vLLM imports)."""

from __future__ import annotations

import contextlib
import gc
import json
import os
import time
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import model_cache_hash
from .env import piecewise_inductor_cache_root
from .piecewise_segments import (
    LAYER_AGNOSTIC_IDX,
    _append_compile_profile,
    piecewise_inductor_artifact_dir,
    piecewise_inductor_package_path,
    piecewise_inductor_shared_cache_dir,
    write_piecewise_inductor_metadata,
)

SEGMENT_MOE = "moe"


def _moe_aot_gc_enabled() -> bool:
    """INFINI_MOE_AOT_GC defaults on; set 0/false/off to disable compile GC."""
    raw = os.environ.get("INFINI_MOE_AOT_GC", "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _release_cuda_compile_state() -> None:
    """Drop compile-time peak: GC + empty_cache (mirrors pre_attn cleanup)."""
    if not _moe_aot_gc_enabled():
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _load_hf_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def moe_routing_hparams(cfg: dict) -> dict:
    n_routed = int(cfg.get("n_routed_experts") or cfg.get("num_experts") or 0)
    return {
        "hidden_size": int(cfg["hidden_size"]),
        "moe_intermediate_size": int(cfg["moe_intermediate_size"]),
        "n_routed_experts": n_routed,
        "num_experts_per_tok": int(cfg["num_experts_per_tok"]),
        "n_group": int(cfg.get("n_group") or 1),
        "topk_group": int(cfg.get("topk_group") or 1),
        "norm_topk_prob": bool(cfg.get("norm_topk_prob", True)),
        "routed_scaling_factor": float(cfg.get("routed_scaling_factor", 1.0)),
        "n_shared_experts": int(cfg.get("n_shared_experts") or 1),
    }


def grouped_sigmoid_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Port of ``minicpm5_moe_router_cpu_detail::run_router_topk_cpu``.

    logits/bias: [T, E] / [E]; returns topk_weights, topk_ids — each [T, K].
    """
    scores = torch.sigmoid(logits.to(dtype=torch.float32))
    bias_f = bias.to(dtype=torch.float32).reshape(1, -1)
    choice = scores + bias_f
    t_tokens, n_experts = choice.shape
    # MiniCPM5 default n_group=1: skip group mask (no-op that still launched topk).
    if n_group != 1:
        experts_per_group = n_experts // n_group
        choice_g = choice.view(t_tokens, n_group, experts_per_group)
        top2 = choice_g.topk(k=min(2, experts_per_group), dim=-1).values
        group_scores = top2.sum(dim=-1)
        group_idx = group_scores.topk(k=topk_group, dim=-1).indices
        keep = torch.zeros(t_tokens, n_group, device=choice.device, dtype=torch.bool)
        keep.scatter_(1, group_idx, True)
        mask = keep.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(
            t_tokens, n_experts
        )
        choice = torch.where(mask, choice, torch.zeros_like(choice))
    topk_vals, topk_ids = torch.topk(choice, k=top_k, dim=-1)
    del topk_vals
    topk_weights = torch.gather(scores, 1, topk_ids)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * float(routed_scaling_factor)
    return topk_weights, topk_ids


def grouped_sigmoid_topk_host(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager decode: D2H logits + NumPy top-k + H2D (n_group=1 only)."""
    import numpy as np

    device = logits.device
    row = logits.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    if row.ndim == 1:
        row = row.reshape(1, -1)
    bias_np = bias.detach().to(dtype=torch.float32, device="cpu").numpy().reshape(-1)
    t_tokens, n_experts = row.shape
    w_out = np.empty((t_tokens, top_k), dtype=np.float32)
    id_out = np.empty((t_tokens, top_k), dtype=np.int32)
    for t in range(t_tokens):
        logits_t = row[t]
        # Stable sigmoid (avoid overflow).
        pos = logits_t >= 0
        z = np.empty_like(logits_t)
        z[pos] = np.exp(-logits_t[pos])
        z[~pos] = np.exp(logits_t[~pos])
        scores = np.empty_like(logits_t)
        scores[pos] = 1.0 / (1.0 + z[pos])
        scores[~pos] = z[~pos] / (1.0 + z[~pos])
        choice = scores + bias_np
        k = min(top_k, n_experts)
        idx = np.argpartition(-choice, k - 1)[:k]
        idx = idx[np.argsort(-choice[idx])]
        w = scores[idx]
        if norm_topk_prob:
            w = w / (w.sum() + 1e-20)
        w = w * float(routed_scaling_factor)
        w_out[t, :k] = w
        id_out[t, :k] = idx.astype(np.int32)
    topk_w = torch.as_tensor(w_out, dtype=torch.float32, device=device)
    topk_ids = torch.as_tensor(id_out, dtype=torch.int32, device=device)
    return topk_w, topk_ids


def _silu_mlp(x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """x [..., H]; w_gate_up [2I, H]; w_down [H, I]."""
    gu = F.linear(x, w_gate_up)
    gate, up = gu.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, w_down)


class PiecewiseMoeSegmentFunctional(nn.Module):
    """Layer-agnostic MoE: hidden + external packed weights → MoE output."""

    def __init__(
        self,
        *,
        bucket: int,
        hidden_size: int,
        moe_intermediate_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        valid_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.bucket = int(bucket)
        self.hidden_size = int(hidden_size)
        self.moe_intermediate_size = int(moe_intermediate_size)
        self.n_routed_experts = int(n_routed_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.n_group = int(n_group)
        self.topk_group = int(topk_group)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.valid_seq_len = int(valid_seq_len) if valid_seq_len is not None else self.bucket
        if self.n_group <= 0 or (self.n_routed_experts % self.n_group) != 0:
            raise ValueError("invalid n_group / n_routed_experts for MoE segment")

    def forward(
        self,
        hidden_states: torch.Tensor,
        gate_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        shared_gate_up: torch.Tensor,
        shared_down: torch.Tensor,
    ) -> torch.Tensor:
        # hidden: [1, B, H] (bucket-padded). When export bakes valid < bucket
        # (decode-oriented), router + fused_moe_routed run on the valid prefix only.
        # Full-bucket export (valid == bucket, the deploy default) is unchanged.
        batch, bucket, hidden = hidden_states.shape
        valid = min(self.valid_seq_len, bucket)
        hs = hidden_states if valid >= bucket else hidden_states.narrow(1, 0, valid)
        x = hs.reshape(batch * valid, hidden)
        logits = F.linear(x.to(dtype=torch.float32), gate_weight.to(dtype=torch.float32))
        topk_w, topk_ids = grouped_sigmoid_topk(
            logits,
            e_score_correction_bias,
            top_k=self.num_experts_per_tok,
            n_group=self.n_group,
            topk_group=self.topk_group,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        routed = torch.ops.infinilm.fused_moe_routed(
            x, topk_w, topk_ids, w_gate_up, w_down
        )
        shared = _silu_mlp(x, shared_gate_up, shared_down)
        out = (routed + shared).view(batch, valid, hidden)
        if valid < bucket:
            out_full = hidden_states.new_zeros(batch, bucket, hidden)
            out_full[:, :valid, :] = out
            return out_full
        return out


def make_moe_example_inputs(
    *,
    bucket: int,
    hidden_size: int,
    moe_intermediate_size: int,
    n_routed_experts: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, ...]:
    """Random tensors matching Functional forward I/O (weights as inputs)."""
    i = int(moe_intermediate_size)
    e = int(n_routed_experts)
    h = int(hidden_size)
    hidden = torch.randn(1, bucket, h, device=device, dtype=dtype)
    gate_weight = torch.randn(e, h, device=device, dtype=dtype)
    bias = torch.randn(e, device=device, dtype=dtype)
    w_gate_up = torch.randn(e, 2 * i, h, device=device, dtype=dtype)
    w_down = torch.randn(e, h, i, device=device, dtype=dtype)
    shared_gate_up = torch.randn(2 * i, h, device=device, dtype=dtype)
    shared_down = torch.randn(h, i, device=device, dtype=dtype)
    return (
        hidden,
        gate_weight,
        bias,
        w_gate_up,
        w_down,
        shared_gate_up,
        shared_down,
    )


def load_moe_weight_example_from_hf(
    model_path: str,
    *,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, ...]:
    """Example weight inputs for export.

    Prefers safetensors shards when present. For single large ``pytorch_model.bin``
    checkpoints (MiniCPM5 ~28GiB), uses random tensors — packages are layer-agnostic
    and real weights are bound at replay via the resolver.
    """
    cfg = _load_hf_config(model_path)
    hp = moe_routing_hparams(cfg)
    i = hp["moe_intermediate_size"]
    e = hp["n_routed_experts"]
    h = hp["hidden_size"]
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    if int(layer_idx) < first_dense:
        raise ValueError(f"layer {layer_idx} is dense (first_k_dense_replace={first_dense})")

    prefix = f"model.layers.{int(layer_idx)}.mlp."
    state: dict[str, torch.Tensor] = {}

    try:
        from safetensors import safe_open

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
            shards: dict[str, list[str]] = {}
            for key, shard in weight_map.items():
                if key.startswith(prefix):
                    shards.setdefault(shard, []).append(key)
            for shard, keys in shards.items():
                with safe_open(os.path.join(model_path, shard), framework="pt") as sf:
                    for key in keys:
                        state[key[len(prefix) :]] = sf.get_tensor(key)
        else:
            single = os.path.join(model_path, "model.safetensors")
            if os.path.isfile(single):
                with safe_open(single, framework="pt") as sf:
                    for key in sf.keys():
                        if key.startswith(prefix):
                            state[key[len(prefix) :]] = sf.get_tensor(key)
    except ImportError:
        state = {}

    if not state:
        _hidden, *weights = make_moe_example_inputs(
            bucket=1,
            hidden_size=h,
            moe_intermediate_size=i,
            n_routed_experts=e,
            device=device,
            dtype=dtype,
        )
        del _hidden
        return tuple(weights)

    def _to(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(device=device, dtype=dtype).contiguous()

    gate_weight = _to(state["gate.weight"])
    bias_key = (
        "e_score_correction_bias"
        if "e_score_correction_bias" in state
        else "gate.e_score_correction_bias"
    )
    bias = _to(state[bias_key])
    w1 = torch.stack([_to(state[f"experts.{j}.gate_proj.weight"]) for j in range(e)], dim=0)
    w3 = torch.stack([_to(state[f"experts.{j}.up_proj.weight"]) for j in range(e)], dim=0)
    w2 = torch.stack([_to(state[f"experts.{j}.down_proj.weight"]) for j in range(e)], dim=0)
    w_gate_up = torch.cat([w1, w3], dim=1)
    del w1, w3
    shared_gate_up = torch.cat(
        [
            _to(state["shared_experts.gate_proj.weight"]),
            _to(state["shared_experts.up_proj.weight"]),
        ],
        dim=0,
    )
    shared_down = _to(state["shared_experts.down_proj.weight"])
    del state
    return gate_weight, bias, w_gate_up, w2, shared_gate_up, shared_down


def aot_compile_minicpm5_moe_segment(
    *,
    model_path: str,
    bucket: int,
    device: torch.device,
    cache_root: Optional[str] = None,
    valid_seq_len: Optional[int] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    layer_idx_for_weights: int = -1,
    require_aot: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Offline AOTInductor package for MiniCPM5 MoE (does not load TorchLlama).

    Requires ``require_aot=True`` (legacy eager / non-AOT MoE packaging removed).
    Routed experts are exported as opaque ``infinilm.fused_moe_routed``.
    """
    if not require_aot:
        raise ValueError(
            "aot_compile_minicpm5_moe_segment requires require_aot=True "
            "(no legacy index_select MoE path)"
        )

    from infinilm.torch_llama.moe_ops import register_fused_moe_routed_op

    from .aot_hpcc_patch import (
        build_aot_inductor_configs,
        hpcc_aot_compile_profile,
        is_hpcc_aot_environment,
        patch_wrapper_sources,
    )

    register_fused_moe_routed_op()

    cfg = _load_hf_config(model_path)
    hp = moe_routing_hparams(cfg)
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    if layer_idx_for_weights < 0:
        layer_idx_for_weights = first_dense

    valid = int(valid_seq_len) if valid_seq_len is not None else int(bucket)
    root = cache_root or piecewise_inductor_cache_root()
    tp_size = max(1, int(tp_size))
    tp_rank = int(tp_rank)

    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    t_total0 = time.perf_counter()
    segment_module = PiecewiseMoeSegmentFunctional(
        bucket=bucket,
        hidden_size=hp["hidden_size"],
        moe_intermediate_size=hp["moe_intermediate_size"],
        n_routed_experts=hp["n_routed_experts"],
        num_experts_per_tok=hp["num_experts_per_tok"],
        n_group=hp["n_group"],
        topk_group=hp["topk_group"],
        norm_topk_prob=hp["norm_topk_prob"],
        routed_scaling_factor=hp["routed_scaling_factor"],
        valid_seq_len=valid,
    ).eval()
    segment_module.requires_grad_(False)

    weight_inputs = load_moe_weight_example_from_hf(
        model_path,
        layer_idx=layer_idx_for_weights,
        device=device,
        dtype=dtype,
    )
    hidden = torch.randn(
        1, bucket, hp["hidden_size"], device=device, dtype=dtype
    )
    example_inputs = tuple(
        t.detach().requires_grad_(False) for t in (hidden,) + tuple(weight_inputs)
    )

    meta_layer_idx = LAYER_AGNOSTIC_IDX
    artifact_dir = os.path.abspath(
        piecewise_inductor_artifact_dir(
            cache_root=root,
            model_path=model_path,
            segment=SEGMENT_MOE,
            layer_idx=meta_layer_idx,
            bucket=bucket,
            tp_size=tp_size,
            tp_rank=tp_rank,
            layer_agnostic=True,
        )
    )
    os.makedirs(artifact_dir, exist_ok=True)
    package_path = os.path.abspath(
        piecewise_inductor_package_path(
            cache_root=root,
            model_path=model_path,
            segment=SEGMENT_MOE,
            layer_idx=meta_layer_idx,
            bucket=bucket,
            tp_size=tp_size,
            tp_rank=tp_rank,
            legacy_fallback=False,
            layer_agnostic=True,
        )
    )

    inductor_cache = piecewise_inductor_shared_cache_dir(
        cache_root=root, model_path=model_path, tp_rank=tp_rank
    )
    os.makedirs(inductor_cache, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.abspath(inductor_cache)

    inductor_configs = None
    if is_hpcc_aot_environment():
        inductor_configs = build_aot_inductor_configs(artifact_dir=artifact_dir)

    with torch.no_grad():
        exported = torch.export.export(segment_module, example_inputs, strict=False)

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

    profile: dict = {
        "segment": SEGMENT_MOE,
        "bucket": int(bucket),
        "layer_agnostic": True,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
    }
    compiled_path = None
    backend = "aot_inductor"
    t0 = time.perf_counter()
    try:
        try:
            compiled_path = _run_aot_package()
            backend = "aot_inductor"
        except Exception as exc:  # noqa: BLE001
            profile["aot_error"] = str(exc)
            if is_hpcc_aot_environment():
                patched = patch_wrapper_sources(inductor_cache)
                if patched > 0:
                    try:
                        compiled_path = _run_aot_package()
                        backend = "aot_inductor"
                    except Exception as retry_exc:  # noqa: BLE001
                        if require_aot:
                            raise RuntimeError(
                                f"MoE AOT compile failed after patch: {retry_exc}"
                            ) from retry_exc
                        raise
                elif require_aot:
                    raise RuntimeError(f"MoE AOT compile failed: {exc}") from exc
                else:
                    raise
            elif require_aot:
                raise RuntimeError(f"MoE AOT compile failed: {exc}") from exc
            else:
                raise
        profile["aot_package_ms"] = (time.perf_counter() - t0) * 1000.0
        profile["total_ms"] = (time.perf_counter() - t_total0) * 1000.0
        profile["backend"] = backend

        meta_path = write_piecewise_inductor_metadata(
            artifact_dir,
            model_path=model_path,
            segment=SEGMENT_MOE,
            layer_idx=meta_layer_idx,
            bucket=bucket,
            extra={
                "package_path": compiled_path,
                "valid_seq_len": valid,
                "hidden_size": hp["hidden_size"],
                "moe_intermediate_size": hp["moe_intermediate_size"],
                "n_routed_experts": hp["n_routed_experts"],
                "dtype": str(dtype),
                "backend": backend,
                "layer_agnostic": True,
                "tp_size": tp_size,
                "tp_rank": tp_rank,
            },
        )
        _append_compile_profile(cache_root=root, model_path=model_path, record=profile)
        return {
            "artifact_dir": artifact_dir,
            "package_path": compiled_path,
            "meta_path": meta_path,
            "backend": backend,
            "profile": profile,
        }
    finally:
        # Release export graphs / example weights before the next bucket.
        try:
            del exported
        except NameError:
            pass
        try:
            del example_inputs
        except NameError:
            pass
        try:
            del weight_inputs
        except NameError:
            pass
        try:
            del hidden
        except NameError:
            pass
        try:
            del segment_module
        except NameError:
            pass
        _release_cuda_compile_state()


def reference_cpu_moe_forward(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    shared_gate_up: torch.Tensor,
    shared_down: torch.Tensor,
    *,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Eager reference matching Functional (for numeric smoke)."""
    mod = PiecewiseMoeSegmentFunctional(
        bucket=hidden.shape[1],
        hidden_size=hidden.shape[2],
        moe_intermediate_size=w_down.shape[-1],
        n_routed_experts=gate_weight.shape[0],
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        valid_seq_len=hidden.shape[1],
    )
    with torch.no_grad():
        return mod(hidden, gate_weight, bias, w_gate_up, w_down, shared_gate_up, shared_down)
