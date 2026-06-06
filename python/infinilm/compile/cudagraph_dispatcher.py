# Copyright (c) 2025, InfiniCore
"""Slim vLLM CudagraphDispatcher vendored for InfiniLM compiled prefill."""

from __future__ import annotations

import logging
from typing import AbstractSet, Optional, Set

from .config import CompiledPrefillConfig
from .env import prefill_cg_baseline_none

logger = logging.getLogger(__name__)


class InfiniCudagraphDispatcher:
    """Placeholder aligned with vLLM CudagraphDispatcher; prefill PIECEWISE only today.

    FULL / ``uniform_decode=True`` decode graph keys are stubbed (empty set) until
    decode CUDA graph capture lands.
    """

    def __init__(self, cfg: CompiledPrefillConfig) -> None:
        from vllm.config import CUDAGraphMode

        self.cfg = cfg
        self.cudagraph_keys: dict[object, Set[object]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }
        self.keys_initialized = False

    def initialize_cudagraph_keys(self, capture_sizes: list[int]) -> None:
        """Register ``BatchDescriptor(num_tokens=bs)`` keys for PIECEWISE prefill."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor

        self.cudagraph_keys[CUDAGraphMode.PIECEWISE].clear()
        self.cudagraph_keys[CUDAGraphMode.FULL].clear()
        for bs in capture_sizes:
            self.cudagraph_keys[CUDAGraphMode.PIECEWISE].add(
                BatchDescriptor(num_tokens=int(bs), uniform_decode=False)
            )
        # TODO(decode): register FULL keys with uniform_decode=True when decode
        # CUDA graphs are captured (vLLM separate_routine decode path).
        self.keys_initialized = True
        logger.info(
            "InfiniCudagraphDispatcher: initialized PIECEWISE keys for buckets %s",
            sorted(int(b) for b in capture_sizes),
        )

    def dispatch(
        self,
        batch_descriptor: object,
        *,
        seq_len: int,
        bucket: int,
        captured_buckets: Optional[AbstractSet[int]] = None,
    ) -> tuple[object, Optional[object]]:
        """Return ``(CUDAGraphMode, BatchDescriptor)`` for bucket-padded prefill."""
        from vllm.config import CUDAGraphMode

        if not self.cfg.use_cudagraph:
            return CUDAGraphMode.NONE, None

        if not self.keys_initialized:
            logger.warning(
                "InfiniCudagraphDispatcher: keys not initialized; eager Inductor"
            )
            return CUDAGraphMode.NONE, None

        capture_set = {int(b) for b in (self.cfg.cudagraph_capture_sizes or ())}
        if int(bucket) not in capture_set:
            return CUDAGraphMode.NONE, None

        if captured_buckets is None:
            captured_buckets = capture_set
        if int(bucket) not in captured_buckets:
            return CUDAGraphMode.NONE, None

        if prefill_cg_baseline_none():
            return CUDAGraphMode.NONE, None

        # vLLM: FULL decode graphs (stub — key set empty today).
        if batch_descriptor in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, batch_descriptor

        non_uniform_key = batch_descriptor.non_uniform
        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, non_uniform_key

        if non_uniform_key in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, non_uniform_key

        return CUDAGraphMode.NONE, None
