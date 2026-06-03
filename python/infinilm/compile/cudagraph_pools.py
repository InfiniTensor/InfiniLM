# Copyright (c) 2025, InfiniCore
"""Persistent input/slot_mapping buffers for piecewise CUDAGraph replay."""

from __future__ import annotations

import logging
from typing import Optional

import infinicore
import torch

from .env import prefill_cg_debug_ptrs_enabled

logger = logging.getLogger(__name__)


def _paged_capture_effective_seq_len(bucket: int, max_seq: int) -> int:
    """Valid slot prefix length at CUDAGraph capture (must be >= any replay ``seq_len`` in bucket)."""
    del max_seq  # ladder is implicit in ``bucket``; kept for call-site stability.
    return bucket


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
