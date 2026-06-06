# Copyright (c) 2025, InfiniCore
"""CUDAGraph replay mode selection for compiled prefill."""

from __future__ import annotations

import logging
from typing import Optional, AbstractSet

import torch

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .env import min_cudagraph_piecewise_bucket

logger = logging.getLogger(__name__)

_MIN_COMPILED_SEQ_LEN = 8


def min_compiled_prefill_seq_len() -> int:
    """Public accessor for server hybrid gating."""
    return _MIN_COMPILED_SEQ_LEN


def _run_compiled_backbone(
    backbone: VllmPrefillBackbone,
    vllm_config,
    cfg: CompiledPrefillConfig,
    input_ids: torch.Tensor,
    *,
    cudagraph_runtime_mode: Optional[object] = None,
) -> torch.Tensor:
    """Invoke compiled backbone; set vLLM forward context when CUDAGraph is on."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context

    num_tokens = int(input_ids.shape[1])
    min_piecewise = min_cudagraph_piecewise_bucket(cfg.cudagraph_capture_sizes)
    if cudagraph_runtime_mode is None:
        capture_set = set(cfg.cudagraph_capture_sizes or ())
        use_cg = (
            cfg.use_cudagraph
            and num_tokens in capture_set
            and num_tokens >= min_piecewise
        )
        if cfg.use_cudagraph and not use_cg:
            logger.info(
                "compiled prefill: bucket %s not in CUDAGraph capture set %s; "
                "eager Inductor replay",
                num_tokens,
                sorted(capture_set),
            )
        mode = CUDAGraphMode.PIECEWISE if use_cg else CUDAGraphMode.NONE
    else:
        mode = cudagraph_runtime_mode
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        cudagraph_runtime_mode=mode,
        batch_descriptor=BatchDescriptor(num_tokens),
    ):
        return backbone(input_ids)


def cudagraph_runtime_mode_for_paged(
    cfg: CompiledPrefillConfig,
    seq_len: int,
    bucket: int,
    *,
    captured_buckets: Optional[AbstractSet[int]] = None,
    dispatcher: Optional[object] = None,
) -> Optional[object]:
    """CUDAGraph replay mode for bucket-padded compiled prefill.

    Thin delegate to :class:`InfiniCudagraphDispatcher` for backward-compatible
    smoke/tests. When ``dispatcher`` is omitted, a temporary dispatcher is used.
    """
    from vllm.forward_context import BatchDescriptor

    from .cudagraph_dispatcher import InfiniCudagraphDispatcher

    d = dispatcher if dispatcher is not None else InfiniCudagraphDispatcher(cfg)
    if not d.keys_initialized and cfg.cudagraph_capture_sizes:
        d.initialize_cudagraph_keys(list(cfg.cudagraph_capture_sizes))
    mode, _ = d.dispatch(
        BatchDescriptor(num_tokens=int(bucket)),
        seq_len=seq_len,
        bucket=bucket,
        captured_buckets=captured_buckets,
    )
    return mode
