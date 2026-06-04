# Copyright (c) 2025, InfiniCore
"""Inductor warmup and piecewise CUDAGraph capture for compiled prefill."""

from __future__ import annotations

import contextlib
import logging
from typing import List, Optional

import torch

from infinilm.torch_llama.kv_paged import PagedPrefillContext, paged_prefill_context

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .cudagraph_pools import (
    _CudagraphInputPool,
    _CudagraphKvStagingPool,
    _CudagraphSlotMappingPool,
    _CudagraphValidLenPool,
    _maybe_log_cg_ptrs,
    _paged_capture_effective_seq_len,
    kv_staging_context,
    valid_seq_len_context,
)
from .cudagraph_runtime import _run_compiled_backbone
from .env import (
    cudagraph_pool_tier_id,
    cudagraph_pool_tier_repr_bucket,
    prefill_cg_kv_outside_graph,
    prefill_cg_pool_tier_isolation,
)

logger = logging.getLogger(__name__)

_TIER_GRAPH_POOLS: dict[str, object] = {}


def refresh_cudagraph_wrapper_refs(holder) -> int:
    """Snapshot live ``CUDAGraphWrapper`` instances after compile/capture."""
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    import gc

    refs: list = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, CUDAGraphWrapper):
                refs.append(obj)
        except ReferenceError:
            continue
    holder._cudagraph_wrappers = refs
    return len(refs)


def _wrapper_refs(backbone: VllmPrefillBackbone) -> list:
    return list(getattr(backbone, "_cudagraph_wrappers", ()))


def _iter_cudagraph_wrappers(root: Optional[torch.nn.Module] = None):
    if root is None:
        return
    yield from _wrapper_refs(root)


def _graph_pool_for_bucket(bucket: int):
    from vllm.platforms import current_platform

    if not prefill_cg_pool_tier_isolation():
        return current_platform.get_global_graph_pool()
    tier = cudagraph_pool_tier_id(bucket)
    pool = _TIER_GRAPH_POOLS.get(tier)
    if pool is None:
        pool = current_platform.graph_pool_handle()
        _TIER_GRAPH_POOLS[tier] = pool
    return pool


def sync_cudagraph_wrapper_pools(backbone: VllmPrefillBackbone, bucket: int) -> None:
    """Point piecewise wrappers at the tier pool for ``bucket`` (when isolation on)."""
    if not prefill_cg_pool_tier_isolation():
        return
    refs = _wrapper_refs(backbone)
    if not refs:
        return
    pool = _graph_pool_for_bucket(bucket)
    for wrapper in refs:
        wrapper.graph_pool = pool


def cudagraph_bucket_has_piecewise_entry(
    backbone: VllmPrefillBackbone,
    bucket: int,
) -> bool:
    """Return True when a piecewise CUDAGraph entry exists for ``bucket``."""
    for wrapper in _iter_cudagraph_wrappers(backbone):
        for desc in wrapper.concrete_cudagraph_entries:
            if int(desc.num_tokens) == int(bucket):
                return True
    return False


def sync_captured_buckets_from_wrappers(
    backbone: VllmPrefillBackbone,
    capture_sizes: Optional[object],
) -> set[int]:
    """Buckets with live piecewise entries (capture succeeded)."""
    capture_set = {int(b) for b in (capture_sizes or ())}
    return {
        b
        for b in capture_set
        if cudagraph_bucket_has_piecewise_entry(backbone, b)
    }


def invalidate_cudagraph_entries(
    backbone: VllmPrefillBackbone,
    buckets: Optional[List[int]] = None,
) -> int:
    """Drop cached piecewise graphs so the next PIECEWISE pass re-captures."""
    bucket_set = None if buckets is None else {int(b) for b in buckets}
    cleared = 0
    for wrapper in _iter_cudagraph_wrappers(backbone):
        if bucket_set is None:
            cleared += len(wrapper.concrete_cudagraph_entries)
            wrapper.concrete_cudagraph_entries.clear()
            continue
        drop = [
            desc
            for desc in wrapper.concrete_cudagraph_entries
            if desc.num_tokens in bucket_set
        ]
        for desc in drop:
            del wrapper.concrete_cudagraph_entries[desc]
        cleared += len(drop)
    if cleared:
        logger.info(
            "compiled prefill: invalidated %s CUDAGraph entries for bucket(s) %s",
            cleared,
            sorted(bucket_set) if bucket_set is not None else "all",
        )
    return cleared


def reset_cudagraph_pools_for_buckets(
    backbone: Optional[VllmPrefillBackbone] = None,
    buckets: Optional[List[int]] = None,
) -> None:
    """Allocate fresh driver graph pool(s) after partial replay poisons memory."""
    from vllm.platforms import current_platform

    if prefill_cg_pool_tier_isolation():
        if buckets:
            tiers = sorted(set(cudagraph_pool_tier_id(int(b)) for b in buckets))
        else:
            tiers = ["short", "mid", "long"]
        for tier in tiers:
            _TIER_GRAPH_POOLS.pop(tier, None)
            _TIER_GRAPH_POOLS[tier] = current_platform.graph_pool_handle()
        if backbone is not None:
            for tier in tiers:
                sync_cudagraph_wrapper_pools(
                    backbone, cudagraph_pool_tier_repr_bucket(tier)
                )
        return
    cls = current_platform.__class__
    pool = current_platform.graph_pool_handle()
    cls._global_graph_pool = pool
    if backbone is not None:
        for wrapper in _iter_cudagraph_wrappers(backbone):
            wrapper.graph_pool = pool


def reset_global_cudagraph_pool(
    backbone: Optional[VllmPrefillBackbone] = None,
) -> None:
    """Backward-compatible alias: reset all pools (or global pool when isolation off)."""
    reset_cudagraph_pools_for_buckets(backbone, buckets=None)


def _warmup_compiled_backbone(
    backbone: VllmPrefillBackbone,
    vllm_config,
    cfg: CompiledPrefillConfig,
    dev: torch.device,
    vocab: int,
    warmup_seq_lens: List[int],
    *,
    kv_staging_pool: Optional[_CudagraphKvStagingPool] = None,
    valid_len_pool: Optional[_CudagraphValidLenPool] = None,
) -> None:
    """Phase 1: Inductor compile only (no stream capture)."""
    from vllm.config import CUDAGraphMode

    for seq_len in warmup_seq_lens:
        dummy = torch.zeros((1, seq_len), dtype=torch.long, device=dev)
        with torch.inference_mode():
            ctx_stack = []
            if kv_staging_pool is not None:
                ctx_stack.append(kv_staging_context(kv_staging_pool, seq_len))
            if valid_len_pool is not None:
                valid_tensor, valid_mask = valid_len_pool.stage(seq_len, seq_len)
                ctx_stack.append(
                    valid_seq_len_context(valid_tensor, mask=valid_mask)
                )
            with contextlib.ExitStack() as stack:
                for cm in ctx_stack:
                    stack.enter_context(cm)
                out = _run_compiled_backbone(
                    backbone,
                    vllm_config,
                    cfg,
                    dummy,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                )
        assert out.shape == (1, seq_len, vocab), out.shape


def _capture_cudagraph_bucket(
    backbone: VllmPrefillBackbone,
    vllm_config,
    cfg: CompiledPrefillConfig,
    dev: torch.device,
    vocab: int,
    seq_len: int,
    input_pool: _CudagraphInputPool,
    *,
    kv_layers: Optional[list] = None,
    slot_mapping_pool: Optional[_CudagraphSlotMappingPool] = None,
    kv_staging_pool: Optional[_CudagraphKvStagingPool] = None,
    valid_len_pool: Optional[_CudagraphValidLenPool] = None,
    block_size: int = 256,
    use_paged_ctx: bool = False,
) -> None:
    """Capture (or re-capture) piecewise CUDAGraph for a single bucket."""
    from vllm.config import CUDAGraphMode

    kv_outside = prefill_cg_kv_outside_graph() and kv_staging_pool is not None
    if kv_outside:
        use_paged_ctx = False

    sync_cudagraph_wrapper_pools(backbone, seq_len)

    num_warmups = vllm_config.compilation_config.cudagraph_num_of_warmups
    dummy = input_pool.get(seq_len)
    dummy.zero_()
    if seq_len > 0:
        dummy[0, :seq_len].copy_(
            torch.arange(1, seq_len + 1, dtype=dummy.dtype, device=dev)
        )
    paged_ctx: Optional[PagedPrefillContext] = None
    if use_paged_ctx:
        if kv_layers is None or slot_mapping_pool is None:
            raise ValueError(
                "paged CUDAGraph capture requires kv_layers and slot_mapping_pool"
            )
        staged_slots = slot_mapping_pool.synthetic(
            seq_len,
            effective_seq_len=_paged_capture_effective_seq_len(
                seq_len, cfg.max_seq_len
            ),
        )
        paged_ctx = PagedPrefillContext(
            kv_layers=list(kv_layers),
            slot_mapping=staged_slots,
            block_size=block_size,
        )
    _maybe_log_cg_ptrs(
        f"capture bucket={seq_len}",
        input_ids=dummy,
        slot_mapping=(
            slot_mapping_pool.get(seq_len)
            if slot_mapping_pool is not None
            else None
        ),
    )

    def _capture_once(cudagraph_runtime_mode) -> torch.Tensor:
        ctx_stack = []
        if kv_outside:
            ctx_stack.append(kv_staging_context(kv_staging_pool, seq_len))
        elif paged_ctx is not None:
            ctx_stack.append(paged_prefill_context(paged_ctx))
        if valid_len_pool is not None:
            valid_tensor, valid_mask = valid_len_pool.stage(seq_len, seq_len)
            ctx_stack.append(
                valid_seq_len_context(valid_tensor, mask=valid_mask)
            )
        with contextlib.ExitStack() as stack:
            for cm in ctx_stack:
                stack.enter_context(cm)
            return _run_compiled_backbone(
                backbone,
                vllm_config,
                cfg,
                dummy,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
            )

    with torch.inference_mode():
        none_runs = max(1, num_warmups)
        for _ in range(none_runs):
            _capture_once(CUDAGraphMode.NONE)
        out = _capture_once(CUDAGraphMode.PIECEWISE)
    assert out.shape == (1, seq_len, vocab), out.shape


def recapture_cudagraph_buckets(
    backbone: VllmPrefillBackbone,
    vllm_config,
    cfg: CompiledPrefillConfig,
    dev: torch.device,
    vocab: int,
    buckets: List[int],
    input_pool: _CudagraphInputPool,
    *,
    kv_layers: Optional[list] = None,
    slot_mapping_pool: Optional[_CudagraphSlotMappingPool] = None,
    kv_staging_pool: Optional[_CudagraphKvStagingPool] = None,
    valid_len_pool: Optional[_CudagraphValidLenPool] = None,
    block_size: int = 256,
    use_paged_ctx: bool = False,
    capture_order: Optional[str] = None,
) -> None:
    """Re-capture piecewise CUDAGraphs after partial-bucket Inductor replay poisons a pool."""
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    if not buckets:
        return
    kv_outside = prefill_cg_kv_outside_graph() and kv_staging_pool is not None
    if kv_outside:
        use_paged_ctx = False
    unique = sorted(set(int(b) for b in buckets))
    if capture_order == "ascending":
        sizes = unique
    elif capture_order == "descending":
        sizes = list(reversed(unique))
    else:
        sizes = sorted(unique, reverse=not use_paged_ctx)
    invalidate_cudagraph_entries(backbone, sizes)
    reset_cudagraph_pools_for_buckets(backbone, sizes)
    logger.info(
        "compiled prefill: re-capturing piecewise CUDAGraphs for bucket(s) %s "
        "(partial replay poison)",
        sizes,
    )
    set_cudagraph_capturing_enabled(True)
    try:
        for seq_len in sizes:
            _capture_cudagraph_bucket(
                backbone,
                vllm_config,
                cfg,
                dev,
                vocab,
                seq_len,
                input_pool,
                kv_layers=kv_layers,
                slot_mapping_pool=slot_mapping_pool,
                kv_staging_pool=kv_staging_pool,
                valid_len_pool=valid_len_pool,
                block_size=block_size,
                use_paged_ctx=use_paged_ctx,
            )
    finally:
        set_cudagraph_capturing_enabled(False)


def _capture_cudagraphs(
    backbone: VllmPrefillBackbone,
    vllm_config,
    cfg: CompiledPrefillConfig,
    dev: torch.device,
    vocab: int,
    capture_sizes: List[int],
    input_pool: _CudagraphInputPool,
    *,
    kv_layers: Optional[list] = None,
    slot_mapping_pool: Optional[_CudagraphSlotMappingPool] = None,
    kv_staging_pool: Optional[_CudagraphKvStagingPool] = None,
    valid_len_pool: Optional[_CudagraphValidLenPool] = None,
    block_size: int = 256,
    use_paged_ctx: bool = False,
) -> None:
    """Phase 2: piecewise CUDAGraph capture on already-compiled bytecode."""
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled

    kv_outside = prefill_cg_kv_outside_graph() and kv_staging_pool is not None
    if kv_outside:
        use_paged_ctx = False

    # Large shapes first so smaller captures reuse the graph memory pool.
    # With paged KV side effects, ascending order avoids corrupting larger graphs.
    sizes = sorted(capture_sizes, reverse=not use_paged_ctx)
    logger.info(
        "compiled prefill: capturing piecewise CUDAGraphs for %s bucket(s)"
        " (paged_ctx=%s kv_outside_graph=%s)",
        len(sizes),
        use_paged_ctx,
        kv_outside,
    )
    set_cudagraph_capturing_enabled(True)
    try:
        for seq_len in sizes:
            _capture_cudagraph_bucket(
                backbone,
                vllm_config,
                cfg,
                dev,
                vocab,
                seq_len,
                input_pool,
                kv_layers=kv_layers,
                slot_mapping_pool=slot_mapping_pool,
                kv_staging_pool=kv_staging_pool,
                valid_len_pool=valid_len_pool,
                block_size=block_size,
                use_paged_ctx=use_paged_ctx,
            )
    finally:
        set_cudagraph_capturing_enabled(False)
