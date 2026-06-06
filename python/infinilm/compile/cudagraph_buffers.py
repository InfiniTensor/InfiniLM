# Copyright (c) 2025, InfiniCore
"""vLLM-style aliased persistent CUDAGraph input buffers (one base ptr per field)."""

from __future__ import annotations

from typing import Optional

import infinicore
import torch

from .env import prefill_cg_scrub_tail


class CudagraphPersistentBuffers:
    """Single max-bucket buffers with prefix views for piecewise CUDAGraph replay.

    Mirrors vLLM ``CpuGpuBuffer`` + ``input_ids.gpu[:num_input_tokens]``: capture
    and replay share the same underlying ``data_ptr``; only the active prefix is
    copied on staging (garbage tail left stale unless ``SCRUB_TAIL=1``).
    """

    def __init__(
        self,
        device: torch.device,
        max_bucket: int,
        *,
        input_dtype: torch.dtype = torch.long,
    ):
        self.max_bucket = int(max_bucket)
        if self.max_bucket <= 0:
            raise ValueError(f"max_bucket must be positive, got {max_bucket}")
        self._device = device
        self.input_ids = torch.zeros(
            (1, self.max_bucket), dtype=input_dtype, device=device
        )
        self.slot_mapping = torch.full(
            (self.max_bucket,), -1, dtype=torch.int64, device=device
        )
        self.valid_seq_len = torch.zeros((), dtype=torch.int32, device=device)

    def view_input_ids(self, bucket: int) -> torch.Tensor:
        if bucket > self.max_bucket:
            raise ValueError(
                f"bucket {bucket} exceeds max_bucket {self.max_bucket}"
            )
        return self.input_ids[:, :bucket]

    def view_slot_mapping(self, bucket: int) -> torch.Tensor:
        if bucket > self.max_bucket:
            raise ValueError(
                f"bucket {bucket} exceeds max_bucket {self.max_bucket}"
            )
        return self.slot_mapping[:bucket]

    def stage_input_ids(
        self, input_ids: torch.Tensor, bucket: int
    ) -> torch.Tensor:
        """Copy ``input_ids`` prefix into the aliased buffer view."""
        seq_len = int(input_ids.shape[1])
        if seq_len > bucket:
            raise ValueError(
                f"seq_len {seq_len} exceeds cudagraph bucket {bucket}"
            )
        view = self.view_input_ids(bucket)
        if prefill_cg_scrub_tail():
            view.zero_()
        if seq_len > 0:
            view[0, :seq_len].copy_(input_ids[0, :seq_len])
        return view

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

    def stage_slot_mapping(
        self,
        slot_mapping,
        bucket: int,
        seq_len: int,
    ) -> infinicore.Tensor:
        """Copy valid slots into ``buf[:seq_len]``; vLLM-style ``-1`` tail on ``[seq_len:bucket]``."""
        if seq_len > bucket:
            raise ValueError(
                f"seq_len {seq_len} exceeds cudagraph bucket {bucket}"
            )
        buf = self.view_slot_mapping(bucket)
        if seq_len > 0:
            sm = self._slot_mapping_to_torch(slot_mapping)
            buf[:seq_len].copy_(sm[:seq_len])
        if seq_len < bucket:
            buf[seq_len:bucket].fill_(-1)
        return infinicore.from_torch(buf)

    def synthetic_slot_mapping(
        self,
        bucket: int,
        effective_seq_len: Optional[int] = None,
    ) -> infinicore.Tensor:
        """Capture-time slots: ``0..effective_seq_len-1`` valid prefix, ``-1`` tail."""
        if effective_seq_len is None:
            effective_seq_len = bucket
        effective_seq_len = min(int(effective_seq_len), bucket)
        buf = self.view_slot_mapping(bucket)
        if effective_seq_len > 0:
            buf[:effective_seq_len].copy_(
                torch.arange(
                    effective_seq_len, dtype=torch.int64, device=self._device
                )
            )
        if effective_seq_len < bucket:
            buf[effective_seq_len:bucket].fill_(-1)
        return infinicore.from_torch(buf)

    def stage_valid_seq_len(self, seq_len: int) -> torch.Tensor:
        """Set scalar ``valid_seq_len`` (scopes active tokens for flash/inductor)."""
        self.valid_seq_len.fill_(int(seq_len))
        return self.valid_seq_len
