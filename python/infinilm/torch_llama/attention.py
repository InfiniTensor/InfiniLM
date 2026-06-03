# Copyright (c) 2025, InfiniCore
"""Flash-attn forward via ``infinilm.prefill_flash_attention`` splitting op."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .ops import register_prefill_flash_attention_op

SPLITTING_FLASH_ATTN_IMPL = "infinilm_prefill_flash"
_ATTENTION_REGISTERED = False


def splitting_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Same contract as ``flash_attention_forward``; flash runs in a custom op."""
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        raise ValueError(
            "splitting_flash_attention_forward does not support output_attentions or head_mask"
        )

    seq_len = query.shape[2]
    if any(dim == 0 for dim in query.shape):
        raise ValueError("FlashAttention does not support zero-sized query dimensions")

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(
                layer.weight.dtype
                for layer in module.modules()
                if isinstance(layer, torch.nn.Linear)
            )
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    from infinilm.compile.env import prefill_cg_kv_outside_graph, prefill_cg_valid_seq_len
    from infinilm.compile.cudagraph_pools import (
        active_kv_staging_context,
        active_valid_seq_len_tensor,
    )
    from .kv_paged import active_paged_prefill_context

    paged_ctx = active_paged_prefill_context()
    staging_ctx = active_kv_staging_context()
    if prefill_cg_kv_outside_graph() and staging_ctx is not None:
        layer_idx = staging_ctx.next_layer_idx()
        torch.ops.infinilm.stage_paged_kv(key, value, layer_idx)
    elif paged_ctx is not None:
        layer_idx = paged_ctx.next_layer_idx()
        torch.ops.infinilm.write_paged_kv(key, value, layer_idx)

    softmax_scale = scaling if scaling is not None else module.scaling
    valid_seq_len = (
        active_valid_seq_len_tensor()
        if prefill_cg_valid_seq_len()
        else None
    )
    attn_output = torch.ops.infinilm.prefill_flash_attention(
        query,
        key,
        value,
        float(softmax_scale),
        bool(is_causal),
        valid_seq_len,
    )
    return attn_output, None


def register_splitting_flash_attention() -> None:
    """Register custom op + attention implementation for compile splitting."""
    global _ATTENTION_REGISTERED
    from .ops import register_stage_paged_kv_op, register_write_paged_kv_op

    register_prefill_flash_attention_op()
    register_write_paged_kv_op()
    register_stage_paged_kv_op()
    if not _ATTENTION_REGISTERED:
        ALL_ATTENTION_FUNCTIONS[SPLITTING_FLASH_ATTN_IMPL] = (
            splitting_flash_attention_forward
        )
        _ATTENTION_REGISTERED = True


def enable_splitting_flash_on_model(model: torch.nn.Module) -> None:
    """Point every decoder layer at the splitting flash attention implementation."""
    register_splitting_flash_attention()
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("expected HuggingFace LlamaForCausalLM-style module")
    for layer in layers:
        layer.self_attn.config._attn_implementation = SPLITTING_FLASH_ATTN_IMPL


def splitting_flash_attention_interface() -> Callable:
    register_splitting_flash_attention()
    return ALL_ATTENTION_FUNCTIONS[SPLITTING_FLASH_ATTN_IMPL]
