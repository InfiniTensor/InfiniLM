# Copyright (c) 2025, InfiniCore
"""Paged KV write path for hybrid compiled prefill ↔ C++ FlashAttentionImpl."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import infinicore
import torch
from infinicore.lib import _infinicore

_TLS = threading.local()


@dataclass
class PagedPrefillContext:
    """Thread-local context consumed by splitting flash attention during prefill."""

    kv_layers: List[object]
    slot_mapping: infinicore.Tensor
    block_size: int = 256
    _layer_idx: int = field(default=0, init=False)

    def next_layer_idx(self) -> int:
        idx = self._layer_idx
        self._layer_idx += 1
        if idx >= len(self.kv_layers):
            raise IndexError(
                f"paged prefill layer index {idx} exceeds kv_layers ({len(self.kv_layers)})"
            )
        return idx


def active_paged_prefill_context() -> Optional[PagedPrefillContext]:
    return getattr(_TLS, "ctx", None)


@contextlib.contextmanager
def paged_prefill_context(ctx: PagedPrefillContext) -> Iterator[PagedPrefillContext]:
    prev = getattr(_TLS, "ctx", None)
    _TLS.ctx = ctx
    try:
        yield ctx
    finally:
        _TLS.ctx = prev


def _cpp_tensor(t) -> object:
    return t._underlying if hasattr(t, "_underlying") else t


def _split_layer_kv(kv_layer):
    """Split per-layer KV ``[2, num_blocks, block_size, num_kv_heads, head_dim]``."""
    kv = _cpp_tensor(kv_layer)
    k_cache = kv.narrow(0, 0, 1).squeeze(0)
    v_cache = kv.narrow(0, 1, 1).squeeze(0)
    return k_cache, v_cache


def write_layer_kv_from_torch(
    ctx: PagedPrefillContext,
    layer_idx: int,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """
    Write prefill K/V into C++ paged cache for ``layer_idx``.

    ``key`` / ``value`` are BHSD ``[batch, seq, num_kv_heads, head_dim]`` (batch=1).
    Matches ``FlashAttentionImpl::do_kv_cache_update`` permute + ``paged_caching_``.
    """
    if key.dim() != 4 or value.dim() != 4:
        raise ValueError("expected key/value rank-4 BHSD tensors")
    seq_len = key.shape[1]
    k_tokens = key.reshape(seq_len, key.shape[2], key.shape[3]).contiguous()
    v_tokens = value.reshape(seq_len, value.shape[2], value.shape[3]).contiguous()

    k_cache, v_cache = _split_layer_kv(ctx.kv_layers[layer_idx])
    k_pool = k_cache.permute([0, 2, 1, 3])
    v_pool = v_cache.permute([0, 2, 1, 3])

    slot_mapping = ctx.slot_mapping
    if hasattr(slot_mapping, "shape") and int(slot_mapping.shape[0]) > seq_len:
        slot_mapping = slot_mapping[:seq_len]
    target_device = infinicore.device("cuda", 0)
    if str(slot_mapping.device) != str(target_device):
        slot_mapping = slot_mapping.to(target_device)

    _infinicore.paged_caching_(
        k_pool,
        v_pool,
        _cpp_tensor(infinicore.from_torch(k_tokens)),
        _cpp_tensor(infinicore.from_torch(v_tokens)),
        _cpp_tensor(slot_mapping),
    )
