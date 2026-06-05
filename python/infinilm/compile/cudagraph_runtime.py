# Copyright (c) 2025, InfiniCore
"""CUDAGraph replay mode selection and bucket priming for compiled prefill."""

from __future__ import annotations

import contextlib
import logging
from typing import Optional, Set, List, AbstractSet

import torch

from infinilm.torch_llama.kv_paged import PagedPrefillContext, paged_prefill_context

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .cudagraph_pools import (
    _CudagraphInputPool,
    _CudagraphKvStagingPool,
    _CudagraphSlotMappingPool,
    _CudagraphValidLenPool,
    _paged_capture_effective_seq_len,
    kv_staging_context,
    valid_seq_len_context,
)
from .env import (
    cudagraph_buckets_needing_recapture,
    cudagraph_poison_ladder_buckets,
    cudagraph_pool_tier_id,
    min_cudagraph_piecewise_bucket,
    prefill_cg_baseline_none,
    prefill_cg_kv_outside_graph,
    prefill_cg_pool_tier_isolation,
    prefill_cg_valid_seq_len,
)

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

    from .cudagraph_capture import sync_cudagraph_wrapper_pools

    num_tokens = int(input_ids.shape[1])
    if cfg.use_cudagraph and prefill_cg_pool_tier_isolation():
        sync_cudagraph_wrapper_pools(backbone, num_tokens)
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
) -> Optional[object]:
    """CUDAGraph replay mode for bucket-padded compiled prefill.

    One mode per bucket: ``PIECEWISE`` when piecewise capture succeeded for
    ``bucket`` (partial pad requires ``valid_seq_len``); otherwise ``NONE``.
    """
    from vllm.config import CUDAGraphMode

    if not cfg.use_cudagraph:
        return None
    capture_set = {int(b) for b in (cfg.cudagraph_capture_sizes or ())}
    if bucket not in capture_set:
        return CUDAGraphMode.NONE

    if captured_buckets is None:
        captured_buckets = capture_set
    if int(bucket) not in captured_buckets:
        return CUDAGraphMode.NONE

    if prefill_cg_baseline_none():
        return CUDAGraphMode.NONE

    from infinilm.compile.env import prefill_cg_allow_partial_pad

    if seq_len < bucket and not prefill_cg_allow_partial_pad():
        return CUDAGraphMode.NONE

    return CUDAGraphMode.PIECEWISE


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
    kv_staging_pool: Optional[_CudagraphKvStagingPool],
    valid_len_pool: Optional[_CudagraphValidLenPool],
    needs_reprime: Set[int],
    runtime_seen: Set[int],
    prime_seq_len: Optional[int] = None,
    reuse_staged_input: bool = False,
) -> bool:
    if (
        not cfg.use_cudagraph
        or input_pool is None
        or slot_pool is None
    ):
        return False
    if bucket in runtime_seen and bucket not in needs_reprime:
        # Capture marks buckets seen at full length; partial replay still needs
        # a bucket-local prime at ``prime_seq_len < bucket`` before PIECEWISE.
        if prime_seq_len is None or int(prime_seq_len) >= bucket:
            return False
    min_piecewise = min_cudagraph_piecewise_bucket(cfg.cudagraph_capture_sizes)
    allow_small_bucket = (
        prefill_cg_valid_seq_len() and bucket in (cfg.cudagraph_capture_sizes or ())
    )
    if (
        bucket < min_piecewise
        and bucket not in needs_reprime
        and not allow_small_bucket
    ):
        return False
    from vllm.config import CUDAGraphMode

    effective_seq = bucket if prime_seq_len is None else int(prime_seq_len)
    if effective_seq > bucket:
        raise ValueError(
            f"prime_seq_len {effective_seq} exceeds cudagraph bucket {bucket}"
        )

    if reuse_staged_input:
        staged_input = input_pool.get(bucket)
    else:
        staged_input = input_pool.get(bucket)
        staged_input.zero_()
    kv_outside = prefill_cg_kv_outside_graph() and kv_staging_pool is not None
    ctx_stack = []
    if kv_outside:
        ctx_stack.append(kv_staging_context(kv_staging_pool, bucket))
    else:
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
        ctx_stack.append(paged_prefill_context(ctx))
    if valid_len_pool is not None:
        valid_tensor, valid_mask = valid_len_pool.stage(effective_seq, bucket)
        ctx_stack.append(
            valid_seq_len_context(valid_tensor, mask=valid_mask)
        )

    with contextlib.ExitStack() as stack:
        for cm in ctx_stack:
            stack.enter_context(cm)
        _run_compiled_backbone(
            backbone,
            vllm_config,
            cfg,
            staged_input,
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        )
    needs_reprime.discard(bucket)
    runtime_seen.add(bucket)
    return True


def _capture_buckets_in_poison_ladder(
    cfg: CompiledPrefillConfig,
    bucket: int,
) -> Set[int]:
    """Buckets that share a CUDAGraph pool tier with ``bucket`` at or above it."""
    return set(
        cudagraph_poison_ladder_buckets(bucket, cfg.cudagraph_capture_sizes)
    )


def buckets_needing_recapture(
    needs_reprime: Set[int],
    bucket: int,
    capture_sizes: Optional[object] = None,
) -> List[int]:
    """Poisoned capture buckets that must be re-captured before replay at ``bucket``."""
    return list(
        cudagraph_buckets_needing_recapture(needs_reprime, bucket, capture_sizes)
    )


def conservative_reprime_before_piecewise(
    cfg: CompiledPrefillConfig,
    bucket: int,
    last_forward_bucket: Optional[int],
    needs_reprime: Set[int],
) -> None:
    """Mark capture buckets for reprime when a smaller bucket ran since last prime.

    Partial or eager Inductor replay at bucket ``L < B`` can poison the shared
    vLLM CUDAGraph pool used by PIECEWISE replay at ``B``.
    """
    if not cfg.use_cudagraph or last_forward_bucket is None:
        return
    if prefill_cg_pool_tier_isolation():
        same_tier = cudagraph_pool_tier_id(last_forward_bucket) == cudagraph_pool_tier_id(
            bucket
        )
        if same_tier and any(
            b in needs_reprime
            for b in _capture_buckets_in_poison_ladder(cfg, last_forward_bucket)
        ):
            # Larger→smaller replay in one tier: pool already poisoned by partial pad.
            return
    if last_forward_bucket >= bucket:
        return
    needs_reprime.update(_capture_buckets_in_poison_ladder(cfg, bucket))


def mark_partial_cudagraph_replay(
    cfg: CompiledPrefillConfig,
    bucket: int,
    seq_len: int,
    needs_reprime: Set[int],
    *,
    cudagraph_runtime_mode: Optional[object] = None,
) -> None:
    """Record pool poisoning after bucket-padded replay (``NONE`` or ``PIECEWISE``).

    Partial pad replay at ``seq_len < bucket`` corrupts same-bucket and larger
    piecewise graphs on MetaX regardless of the runtime CUDAGraph mode used.
    """
    del cudagraph_runtime_mode  # reserved for debug logging
    if seq_len >= bucket or not cfg.use_cudagraph:
        return
    needs_reprime.update(_capture_buckets_in_poison_ladder(cfg, bucket))
