# Copyright (c) 2025, InfiniCore
"""CUDAGraph replay mode selection and bucket priming for compiled prefill."""

from __future__ import annotations

import logging
from typing import Optional, Set

import torch

from infinilm.torch_llama.kv_paged import PagedPrefillContext, paged_prefill_context

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .cudagraph_pools import (
    _CudagraphInputPool,
    _CudagraphSlotMappingPool,
    _paged_capture_effective_seq_len,
)

logger = logging.getLogger(__name__)

# Partial PIECEWISE replay below this bucket poisons larger CUDAGraphs on MetaX.
PARTIAL_CG_PIECEWISE_MIN_BUCKET = 4096
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
    if cudagraph_runtime_mode is None:
        capture_set = set(cfg.cudagraph_capture_sizes or ())
        use_cg = (
            cfg.use_cudagraph
            and num_tokens in capture_set
            and num_tokens >= PARTIAL_CG_PIECEWISE_MIN_BUCKET
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
) -> Optional[object]:
    """CUDAGraph replay mode for bucket-padded compiled prefill.

    * ``seq_len < bucket``: always ``NONE`` (no partial PIECEWISE; MetaX ATU).
    * ``seq_len == bucket`` and bucket ≥ 4096: ``PIECEWISE`` full-bucket replay.
    * Repro: ``INFINI_PREFILL_CG_ALLOW_PARTIAL=1`` restores auto PIECEWISE on partial.
    """
    from .env import prefill_cg_allow_partial_pad
    from vllm.config import CUDAGraphMode

    if not cfg.use_cudagraph:
        return None
    if prefill_cg_allow_partial_pad():
        return None
    if seq_len < bucket:
        return CUDAGraphMode.NONE
    if seq_len == bucket and bucket >= PARTIAL_CG_PIECEWISE_MIN_BUCKET:
        return CUDAGraphMode.PIECEWISE
    return CUDAGraphMode.NONE


def prime_cudagraph_bucket_runtime(
    *,
    cfg: CompiledPrefillConfig,
    backbone: VllmPrefillBackbone,
    vllm_config,
    bucket: int,
    kv_layers,
    block_size: int,
    input_pool: _CudagraphInputPool,
    slot_pool: _CudagraphSlotMappingPool,
    needs_reprime: Set[int],
    runtime_seen: Set[int],
) -> bool:
    if (
        not cfg.use_cudagraph
        or input_pool is None
        or slot_pool is None
    ):
        return False
    if bucket in runtime_seen and bucket not in needs_reprime:
        return False
    if (
        bucket < PARTIAL_CG_PIECEWISE_MIN_BUCKET
        and bucket not in needs_reprime
    ):
        return False
    from vllm.config import CUDAGraphMode

    dummy = input_pool.get(bucket)
    dummy.zero_()
    staged_slots = slot_pool.synthetic(
        bucket,
        effective_seq_len=_paged_capture_effective_seq_len(
            bucket, cfg.max_seq_len
        ),
    )
    ctx = PagedPrefillContext(
        kv_layers=list(kv_layers),
        slot_mapping=staged_slots,
        block_size=block_size,
    )
    with paged_prefill_context(ctx):
        _run_compiled_backbone(
            backbone,
            vllm_config,
            cfg,
            dummy,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        )
    needs_reprime.discard(bucket)
    runtime_seen.add(bucket)
    return True


def mark_partial_cudagraph_replay(
    cfg: CompiledPrefillConfig,
    bucket: int,
    seq_len: int,
    needs_reprime: Set[int],
) -> None:
    if seq_len >= bucket or not cfg.use_cudagraph:
        return
    capture_set = set(cfg.cudagraph_capture_sizes or ())
    for b in capture_set:
        if b > bucket:
            needs_reprime.add(b)
