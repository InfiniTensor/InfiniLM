# Copyright (c) 2025, InfiniCore
"""Persistent input/slot_mapping buffers for piecewise CUDAGraph replay."""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional

import infinicore
import torch

from .env import prefill_cg_debug_ptrs_enabled

logger = logging.getLogger(__name__)


def _paged_capture_effective_seq_len(bucket: int, max_seq: int) -> int:
    """Valid slot prefix length at CUDAGraph capture (must be >= any replay ``seq_len`` in bucket)."""
    del max_seq  # ladder is implicit in ``bucket``; kept for call-site stability.
    return bucket


class _CudagraphValidSeqLenPool:
    """Scalar int32 ``valid_seq_len`` per bucket (stable ``data_ptr`` for partial CG replay)."""

    def __init__(
        self,
        device: torch.device,
        *,
        capture_buckets: Optional[list[int]] = None,
    ):
        self._device = device
        self._buffers: dict[int, torch.Tensor] = {}
        if capture_buckets:
            for bucket in capture_buckets:
                self.get(bucket)

    def get(self, bucket: int) -> torch.Tensor:
        buf = self._buffers.get(bucket)
        if buf is None:
            buf = torch.tensor([bucket], dtype=torch.int32, device=self._device)
            self._buffers[bucket] = buf
        return buf

    def stage(self, seq_len: int, bucket: int) -> torch.Tensor:
        """Set ``valid_seq_len`` before capture (``seq_len==bucket``) or partial replay."""
        if seq_len > bucket:
            raise ValueError(f"valid_seq_len {seq_len} exceeds bucket {bucket}")
        buf = self.get(bucket)
        buf.fill_(seq_len)
        return buf


class _CudagraphValidMaskPool:
    """Per-bucket ``[1, bucket, 1]`` multiplicative mask (1=valid token, 0=pad tail)."""

    def __init__(
        self,
        device: torch.device,
        *,
        dtype: torch.dtype = torch.float32,
        capture_buckets: Optional[list[int]] = None,
    ):
        self._device = device
        self._dtype = dtype
        self._buffers: dict[int, torch.Tensor] = {}
        if capture_buckets:
            for bucket in capture_buckets:
                self.get(bucket)

    def get(self, bucket: int) -> torch.Tensor:
        buf = self._buffers.get(bucket)
        if buf is None:
            buf = torch.ones((1, bucket, 1), dtype=self._dtype, device=self._device)
            self._buffers[bucket] = buf
        return buf

    def stage(self, seq_len: int, bucket: int) -> torch.Tensor:
        if seq_len > bucket:
            raise ValueError(f"valid mask seq_len {seq_len} exceeds bucket {bucket}")
        buf = self.get(bucket)
        buf.zero_()
        if seq_len > 0:
            buf[:, :seq_len, :].fill_(1)
        return buf


class _CudagraphValidLenPool:
    """Pooled ``valid_seq_len`` scalar + per-token mask for partial CG replay."""

    def __init__(
        self,
        device: torch.device,
        *,
        mask_dtype: torch.dtype = torch.float32,
        capture_buckets: Optional[list[int]] = None,
    ):
        self._seq = _CudagraphValidSeqLenPool(
            device, capture_buckets=capture_buckets
        )
        self._mask = _CudagraphValidMaskPool(
            device, dtype=mask_dtype, capture_buckets=capture_buckets
        )

    def stage(self, seq_len: int, bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._seq.stage(seq_len, bucket),
            self._mask.stage(seq_len, bucket),
        )


@dataclass
class _ValidSeqLenContext:
    tensor: torch.Tensor
    mask: Optional[torch.Tensor] = None


def active_valid_seq_len_tensor() -> Optional[torch.Tensor]:
    ctx = getattr(_TLS, "valid_seq_len_ctx", None)
    return ctx.tensor if ctx is not None else None


def active_valid_seq_mask() -> Optional[torch.Tensor]:
    ctx = getattr(_TLS, "valid_seq_len_ctx", None)
    return ctx.mask if ctx is not None else None


@contextlib.contextmanager
def valid_seq_len_context(
    tensor: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> Iterator[_ValidSeqLenContext]:
    ctx = _ValidSeqLenContext(tensor=tensor, mask=mask)
    prev = getattr(_TLS, "valid_seq_len_ctx", None)
    _TLS.valid_seq_len_ctx = ctx
    try:
        yield ctx
    finally:
        _TLS.valid_seq_len_ctx = prev


class _CudagraphInputPool:
    """Persistent ``input_ids`` buffers for vLLM piecewise CUDAGraph replay.

    vLLM ``CUDAGraphWrapper`` replays without copying runtime inputs; capture and
    replay must use the same ``data_ptr`` per bucket (see vLLM ``gpu_model_runner``).
    """

    def __init__(self, device: torch.device, *, dtype: torch.dtype = torch.long):
        self._device = device
        self._dtype = dtype
        self._buffers: dict[int, torch.Tensor] = {}

    def get(self, bucket: int) -> torch.Tensor:
        buf = self._buffers.get(bucket)
        if buf is None:
            buf = torch.zeros((1, bucket), dtype=self._dtype, device=self._device)
            self._buffers[bucket] = buf
        return buf

    def stage(self, input_ids: torch.Tensor, bucket: int) -> torch.Tensor:
        """Copy tokens into the capture buffer; zero-fill padding tail."""
        buf = self.get(bucket)
        seq_len = int(input_ids.shape[1])
        if seq_len > bucket:
            raise ValueError(f"seq_len {seq_len} exceeds cudagraph bucket {bucket}")
        buf.zero_()
        if seq_len > 0:
            buf[0, :seq_len].copy_(input_ids[0, :seq_len])
        return buf


class _CudagraphSlotMappingPool:
    """Persistent ``slot_mapping`` buffers for paged KV writes during CUDAGraph replay.

    Padding tail is filled with ``-1`` (ignored by ``paged_caching``). Capture and
    replay must use the same underlying buffer per bucket.
    """

    def __init__(self, device: torch.device):
        self._device = device
        self._buffers: dict[int, torch.Tensor] = {}
        self._infini: dict[int, infinicore.Tensor] = {}

    def get(self, bucket: int) -> torch.Tensor:
        buf = self._buffers.get(bucket)
        if buf is None:
            buf = torch.full((bucket,), -1, dtype=torch.int64, device=self._device)
            self._buffers[bucket] = buf
            self._infini[bucket] = infinicore.from_torch(buf)
        return buf

    def _slot_mapping_to_torch(self, slot_mapping) -> torch.Tensor:
        from infinilm.torch_llama.kv_paged import ensure_hybrid_prefill_gpu_context

        ensure_hybrid_prefill_gpu_context(device_index=int(self._device.index))
        if isinstance(slot_mapping, torch.Tensor):
            sm = slot_mapping
        else:
            sm = infinicore.to_torch(slot_mapping.contiguous())
        if sm.device != self._device:
            sm = sm.to(self._device)
        return sm.reshape(-1)

    def stage(
        self,
        slot_mapping,
        bucket: int,
        seq_len: int,
    ) -> infinicore.Tensor:
        """Copy valid slots into ``buf[:seq_len]``; pad ``buf[seq_len:bucket]`` with ``-1``."""
        if seq_len > bucket:
            raise ValueError(f"seq_len {seq_len} exceeds cudagraph bucket {bucket}")
        buf = self.get(bucket)
        buf.fill_(-1)
        if seq_len > 0:
            sm = self._slot_mapping_to_torch(slot_mapping)
            buf[:seq_len].copy_(sm[:seq_len])
        return self._infini[bucket]

    def synthetic(self, bucket: int, effective_seq_len: Optional[int] = None) -> infinicore.Tensor:
        """Capture-time slots matching replay padding (valid prefix, ``-1`` tail)."""
        if effective_seq_len is None:
            effective_seq_len = bucket
        effective_seq_len = min(int(effective_seq_len), bucket)
        buf = self.get(bucket)
        buf.fill_(-1)
        if effective_seq_len > 0:
            buf[:effective_seq_len].copy_(
                torch.arange(effective_seq_len, dtype=torch.int64, device=self._device)
            )
        return self._infini[bucket]


_TLS = threading.local()


@dataclass
class _KvStagingContext:
    """Thread-local staging target for ``infinilm.stage_paged_kv`` during CG replay."""

    pool: "_CudagraphKvStagingPool"
    bucket: int
    _layer_idx: int = field(default=0, init=False)

    def next_layer_idx(self) -> int:
        idx = self._layer_idx
        self._layer_idx += 1
        if idx >= self.pool.num_layers:
            raise IndexError(
                f"kv staging layer index {idx} exceeds num_layers ({self.pool.num_layers})"
            )
        return idx


def active_kv_staging_context() -> Optional[_KvStagingContext]:
    return getattr(_TLS, "ctx", None)


@contextlib.contextmanager
def kv_staging_context(
    pool: "_CudagraphKvStagingPool", bucket: int
) -> Iterator[_KvStagingContext]:
    ctx = _KvStagingContext(pool=pool, bucket=bucket)
    prev = getattr(_TLS, "ctx", None)
    _TLS.ctx = ctx
    try:
        yield ctx
    finally:
        _TLS.ctx = prev


class _CudagraphKvStagingPool:
    """Per-bucket, per-layer K/V buffers for graph-safe staging (stable ``data_ptr``)."""

    def __init__(
        self,
        device: torch.device,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        capture_buckets: Optional[list[int]] = None,
    ):
        self._device = device
        self.num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype = dtype
        self._k: dict[int, list[torch.Tensor]] = {}
        self._v: dict[int, list[torch.Tensor]] = {}
        if capture_buckets:
            for bucket in capture_buckets:
                self._ensure_bucket(bucket)

    def _ensure_bucket(
        self,
        bucket: int,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> None:
        if bucket in self._k:
            return
        shape = (1, bucket, self._num_kv_heads, self._head_dim)
        self._k[bucket] = [
            torch.empty(shape, dtype=self._dtype, device=self._device)
            for _ in range(self.num_layers)
        ]
        self._v[bucket] = [
            torch.empty(shape, dtype=self._dtype, device=self._device)
            for _ in range(self.num_layers)
        ]

    def stage_layer(
        self,
        bucket: int,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        valid_seq_len: Optional[int] = None,
    ) -> None:
        """Copy projected K/V into the persistent staging buffer for ``layer_idx``."""
        self._ensure_bucket(bucket)
        seq_len = int(key.shape[1])
        if valid_seq_len is not None:
            seq_len = min(int(valid_seq_len), seq_len)
        self._k[bucket][layer_idx].zero_()
        self._v[bucket][layer_idx].zero_()
        if seq_len > 0:
            self._k[bucket][layer_idx][:, :seq_len].copy_(key[:, :seq_len])
            self._v[bucket][layer_idx][:, :seq_len].copy_(value[:, :seq_len])

    def staged_layer_kv(
        self, bucket: int, layer_idx: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return staged K/V truncated to the valid token prefix."""
        if bucket not in self._k:
            raise RuntimeError(f"kv staging bucket={bucket} was never primed")
        return (
            self._k[bucket][layer_idx][:, :seq_len],
            self._v[bucket][layer_idx][:, :seq_len],
        )


def _maybe_log_cg_ptrs(
    label: str,
    *,
    input_ids: Optional[torch.Tensor] = None,
    slot_mapping: Optional[torch.Tensor] = None,
) -> None:
    if not prefill_cg_debug_ptrs_enabled():
        return
    parts = [label]
    if input_ids is not None:
        parts.append(f"input_ids ptr={input_ids.data_ptr()}")
    if slot_mapping is not None:
        parts.append(f"slot_mapping ptr={slot_mapping.data_ptr()}")
    logger.info("cudagraph ptrs: %s", " ".join(parts))
