# Copyright (c) 2025, InfiniCore
"""Custom ops for torch.compile splitting boundaries (flash-attn outside Inductor)."""

from __future__ import annotations

from typing import Optional

import torch

_LIB: Optional[torch.library.Library] = None
_REGISTERED = False

PREFILL_FLASH_ATTN_OP = "infinilm.prefill_flash_attention"


def _run_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    from transformers.integrations.flash_attention import (
        _flash_attention_forward,
        _use_top_left_mask,
    )

    return _flash_attention_forward(
        query,
        key,
        value,
        attention_mask=None,
        query_length=query.shape[1],
        is_causal=is_causal,
        dropout=0.0,
        softmax_scale=softmax_scale,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=None,
        implementation="flash_attention_2",
    )


def register_prefill_flash_attention_op() -> None:
    """Idempotent registration of ``infinilm.prefill_flash_attention``."""
    global _LIB, _REGISTERED
    if _REGISTERED:
        return

    _LIB = torch.library.Library("infinilm", "FRAGMENT")
    _LIB.define(
        "prefill_flash_attention("
        "Tensor query, Tensor key, Tensor value, float softmax_scale, bool is_causal"
        ") -> Tensor"
    )

    @torch.library.impl("infinilm::prefill_flash_attention", "CUDA")
    def _prefill_flash_cuda(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float,
        is_causal: bool,
    ) -> torch.Tensor:
        return _run_flash_attention(query, key, value, softmax_scale, is_causal)

    @torch.library.impl("infinilm::prefill_flash_attention", "PrivateUse1")
    def _prefill_flash_maca(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float,
        is_causal: bool,
    ) -> torch.Tensor:
        return _run_flash_attention(query, key, value, softmax_scale, is_causal)

    @torch.library.register_fake("infinilm::prefill_flash_attention")
    def _prefill_flash_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale: float,
        is_causal: bool,
    ) -> torch.Tensor:
        return torch.empty_like(query)

    _REGISTERED = True
