# Copyright (c) 2025, InfiniCore
"""Inductor warmup and piecewise CUDAGraph capture for compiled prefill."""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from infinilm.torch_llama.kv_paged import PagedPrefillContext, paged_prefill_context

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .cudagraph_pools import (
    _CudagraphInputPool,
    _CudagraphSlotMappingPool,
    _maybe_log_cg_ptrs,
    _paged_capture_effective_seq_len,
)
from .cudagraph_runtime import _run_compiled_backbone

logger = logging.getLogger(__name__)


def _warmup_compiled_backbone(
    backbone: VllmPrefillBackbone,
    vllm_config,
    cfg: CompiledPrefillConfig,
    dev: torch.device,
    vocab: int,
    warmup_seq_lens: List[int],
) -> None:
    """Phase 1: Inductor compile only (no stream capture)."""
    from vllm.config import CUDAGraphMode

    for seq_len in warmup_seq_lens:
        dummy = torch.zeros((1, seq_len), dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = _run_compiled_backbone(
                backbone,
                vllm_config,
                cfg,
                dummy,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
        assert out.shape == (1, seq_len, vocab), out.shape


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
    block_size: int = 256,
    use_paged_ctx: bool = False,
) -> None:
    """Phase 2: piecewise CUDAGraph capture on already-compiled bytecode."""
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode

    # Large shapes first so smaller captures reuse the graph memory pool.
    # With paged KV side effects, ascending order avoids corrupting larger graphs.
    sizes = sorted(capture_sizes, reverse=not use_paged_ctx)
    num_warmups = vllm_config.compilation_config.cudagraph_num_of_warmups
    logger.info(
        "compiled prefill: capturing piecewise CUDAGraphs for %s bucket(s)"
        " (paged_ctx=%s)",
        len(sizes),
        use_paged_ctx,
    )
    set_cudagraph_capturing_enabled(True)
    try:
        for seq_len in sizes:
            # Reuse persistent buffers so replay sees the same input addresses.
            dummy = input_pool.get(seq_len)
            dummy.zero_()
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
                if paged_ctx is not None:
                    with paged_prefill_context(paged_ctx):
                        return _run_compiled_backbone(
                            backbone,
                            vllm_config,
                            cfg,
                            dummy,
                            cudagraph_runtime_mode=cudagraph_runtime_mode,
                        )
                return _run_compiled_backbone(
                    backbone,
                    vllm_config,
                    cfg,
                    dummy,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                )

            with torch.inference_mode():
                # At least one NONE replay on the capture path before graph capture
                # (vLLM ``_capture_cudagraphs``; avoids legacy↔capture stream deps).
                none_runs = max(1, num_warmups)
                for _ in range(none_runs):
                    _capture_once(CUDAGraphMode.NONE)
                out = _capture_once(CUDAGraphMode.PIECEWISE)
            assert out.shape == (1, seq_len, vocab), out.shape
    finally:
        set_cudagraph_capturing_enabled(False)
