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


def _to_torch_view(t) -> torch.Tensor:
    if isinstance(t, torch.Tensor):
        return t
    if hasattr(t, "_underlying"):
        return infinicore.to_torch(t)
    fn = getattr(_infinicore, "_tensor_as_torch", None)
    if fn is None:
        raise RuntimeError(
            "infinicore tensor views require InfiniCore built with aten enabled"
        )
    return fn(t)


def ensure_hybrid_prefill_gpu_context(*, device_index: int = 0) -> None:
    """Align InfiniCore + torch CUDA on the AsyncLLMEngine step thread.

    ``basic_llm_processor.build_model_inputs`` uses ``infinicore.from_list`` (CPU)
    before hybrid compiled prefill; without resetting device, MetaX torch GEMM in
    the compiled backbone can ATU-fault on the first server warmup request.
    """
    infinicore.set_device(infinicore.device("cuda", device_index))
    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)


def _ensure_infinicore_gpu_context(torch_tensor: torch.Tensor) -> None:
    """Align thread-local InfiniCore device with torch CUDA before infiniop."""
    if not isinstance(torch_tensor, torch.Tensor) or not torch_tensor.is_cuda:
        return
    ensure_hybrid_prefill_gpu_context(device_index=int(torch_tensor.device.index))


def _slot_mapping_for_caching(slot_mapping, seq_len: int) -> object:
    """Return C++ slot_mapping prefix ``[:seq_len]`` on GPU for ``paged_caching_``."""
    if isinstance(slot_mapping, torch.Tensor):
        sm = slot_mapping.reshape(-1)
    else:
        sm = _to_torch_view(slot_mapping).reshape(-1)
    if sm.shape[0] > seq_len:
        sm = sm[:seq_len]
    if sm.device.type != "cuda":
        sm = sm.to("cuda")
    return _cpp_tensor(infinicore.from_torch(sm.contiguous()))


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
    _ensure_infinicore_gpu_context(key)
    seq_len = key.shape[1]
    k_tokens = (
        key.reshape(seq_len, key.shape[2], key.shape[3]).contiguous().clone()
    )
    v_tokens = (
        value.reshape(seq_len, value.shape[2], value.shape[3]).contiguous().clone()
    )

    k_cache, v_cache = _split_layer_kv(ctx.kv_layers[layer_idx])
    k_pool = k_cache.permute([0, 2, 1, 3])
    v_pool = v_cache.permute([0, 2, 1, 3])

    slot_mapping = _slot_mapping_for_caching(ctx.slot_mapping, seq_len)

    _infinicore.paged_caching_(
        k_pool,
        v_pool,
        _cpp_tensor(infinicore.from_torch(k_tokens)),
        _cpp_tensor(infinicore.from_torch(v_tokens)),
        slot_mapping,
    )


def flush_staged_kv_to_paged_cache(
    staging_pool,
    bucket: int,
    seq_len: int,
    ctx: PagedPrefillContext,
) -> None:
    """Eager post-replay flush: write staged K/V into C++ paged cache per layer."""
    for layer_idx in range(staging_pool.num_layers):
        key, value = staging_pool.staged_layer_kv(bucket, layer_idx, seq_len)
        write_layer_kv_from_torch(ctx, layer_idx, key, value)


def read_paged_kv_at_slots(
    kv_layer,
    slot_indices: list[int],
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V vectors at physical paged slots for layer ``kv_layer`` (batch=1)."""
    k_cache, v_cache = _split_layer_kv(kv_layer)
    k_t = _to_torch_view(k_cache).contiguous()
    v_t = _to_torch_view(v_cache).contiguous()
    k_rows: list[torch.Tensor] = []
    v_rows: list[torch.Tensor] = []
    for slot in slot_indices:
        block = int(slot) // block_size
        offset = int(slot) % block_size
        k_rows.append(k_t[block, offset])
        v_rows.append(v_t[block, offset])
    return torch.stack(k_rows, dim=0), torch.stack(v_rows, dim=0)
