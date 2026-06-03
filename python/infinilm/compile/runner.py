# Copyright (c) 2025, InfiniCore
"""Init-compile torch prefill backbone with vLLM ``VllmBackend`` + ``CompilationConfig``."""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .cudagraph_capture import _capture_cudagraphs, _warmup_compiled_backbone
from .cudagraph_pools import (
    _CudagraphInputPool,
    _CudagraphSlotMappingPool,
    _maybe_log_cg_ptrs,
)
from .cudagraph_runtime import (
    cudagraph_runtime_mode_for_paged,
    mark_partial_cudagraph_replay,
    min_compiled_prefill_seq_len,
    prime_cudagraph_bucket_runtime,
    _run_compiled_backbone,
)
from .env import compile_bucket_mode, compile_buckets

logger = logging.getLogger(__name__)

__all__ = [
    "CompiledPrefillRunner",
    "compile_prefill_backbone",
    "min_compiled_prefill_seq_len",
]


def _snapshot_gpu_mem(checkpoint: str, *, once: bool = False) -> None:
    from .mem_profile import snapshot_gpu_mem

    snapshot_gpu_mem(checkpoint, once=once)


def _build_vllm_config(cfg: CompiledPrefillConfig):
    from vllm.config import (
        CompilationConfig,
        CompilationLevel,
        CUDAGraphMode,
        ModelConfig,
        PassConfig,
        VllmConfig,
    )

    model_config = ModelConfig(
        model=cfg.model_path,
        trust_remote_code=True,
        max_model_len=cfg.max_seq_len,
    )
    cudagraph_mode = (
        CUDAGraphMode.NONE if not cfg.use_cudagraph else CUDAGraphMode.PIECEWISE
    )
    pass_config = PassConfig(enable_fusion=cfg.enable_fusion)
    compilation_config = CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        cache_dir=cfg.cache_dir,
        compile_sizes=list(cfg.compile_sizes),
        cudagraph_mode=cudagraph_mode,
        use_inductor=True,
        splitting_ops=list(cfg.splitting_ops),
        pass_config=pass_config,
    )
    compilation_config.init_with_cudagraph_sizes(list(cfg.cudagraph_capture_sizes))
    return VllmConfig(model_config=model_config, compilation_config=compilation_config)


def compile_prefill_backbone(
    cfg: CompiledPrefillConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    warmup_seq_lens: Optional[List[int]] = None,
    cpp_state_dict: Optional[dict] = None,
    kv_layers: Optional[list] = None,
    block_size: int = 256,
    defer_cudagraph_capture: bool = False,
) -> Tuple[
    VllmPrefillBackbone,
    float,
    object,
    Optional[_CudagraphInputPool],
    Optional[_CudagraphSlotMappingPool],
]:
    """
    Build ``VllmPrefillBackbone`` (vLLM torch.compile wrapper + ``VllmBackend``).

    Returns:
        (backbone, compile_seconds, vllm_config, input_pool, slot_mapping_pool)
        ``input_pool`` / ``slot_mapping_pool`` when ``cfg.use_cudagraph``.
    """
    from vllm.config import set_current_vllm_config

    cfg.write_metadata()
    os.makedirs(cfg.cache_dir, exist_ok=True)

    vllm_config = _build_vllm_config(cfg)

    logger.info(
        "Compiling graph for compile range (1, %s]; cache_dir=%s",
        cfg.max_seq_len,
        cfg.cache_dir,
    )
    if cfg.use_cudagraph:
        logger.info(
            "compiled prefill: piecewise CUDAGraph enabled (INFINI_PREFILL_CUDAGRAPH=1) "
            "capture_sizes=%s",
            cfg.cudagraph_capture_sizes,
        )

    if warmup_seq_lens is None:
        warmup_seq_lens = [8]

    t0 = time.perf_counter()
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=cfg.prefix):
        backbone = VllmPrefillBackbone(
            vllm_config,
            cfg.model_path,
            prefix=cfg.prefix,
            device=device,
            dtype=dtype,
            cpp_state_dict=cpp_state_dict,
        )
        dev = next(backbone.inner.parameters()).device
        vocab = backbone.inner.config.vocab_size
        _snapshot_gpu_mem("T0_model_load")
        input_pool: Optional[_CudagraphInputPool] = None
        slot_pool: Optional[_CudagraphSlotMappingPool] = None
        use_paged_capture = cpp_state_dict is not None
        if cfg.use_cudagraph:
            input_pool = _CudagraphInputPool(dev)
            if use_paged_capture:
                slot_pool = _CudagraphSlotMappingPool(dev)
            _warmup_compiled_backbone(
                backbone, vllm_config, cfg, dev, vocab, warmup_seq_lens
            )
            _snapshot_gpu_mem("T1_after_inductor_warmup")
            if not defer_cudagraph_capture and (
                not use_paged_capture or kv_layers is not None
            ):
                capture_sizes = list(cfg.cudagraph_capture_sizes)
                _capture_cudagraphs(
                    backbone,
                    vllm_config,
                    cfg,
                    dev,
                    vocab,
                    capture_sizes,
                    input_pool,
                    kv_layers=kv_layers,
                    slot_mapping_pool=slot_pool,
                    block_size=block_size,
                    use_paged_ctx=use_paged_capture and kv_layers is not None,
                )
                _snapshot_gpu_mem("T2_after_cudagraph_capture")
            elif defer_cudagraph_capture:
                logger.info(
                    "compiled prefill: deferring CUDAGraph capture until kv_layers "
                    "are available (share_weights + paged KV path)"
                )
        else:
            _warmup_compiled_backbone(
                backbone, vllm_config, cfg, dev, vocab, warmup_seq_lens
            )
            _snapshot_gpu_mem("T1_after_inductor_warmup")

    elapsed = time.perf_counter() - t0
    vllm_config.compilation_config.compilation_time = elapsed
    logger.info("torch.compile takes %.3f s", elapsed)
    return backbone, elapsed, vllm_config, input_pool, slot_pool


class CompiledPrefillRunner:
    """Loads torch backbone, runs init compile, exposes compiled prefill forward."""

    _REPLAY_CHECKPOINTS = (
        (4096, "T4_first_4096_replay"),
        (8192, "T5_first_8192_replay"),
    )

    def __init__(
        self,
        cfg: CompiledPrefillConfig,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        warmup_seq_lens: Optional[List[int]] = None,
        cpp_state_dict: Optional[dict] = None,
        kv_layers: Optional[list] = None,
        block_size: int = 256,
    ):
        self.cfg = cfg
        self._block_size = block_size
        self._cudagraph_capture_done = False
        if warmup_seq_lens is None:
            warmup_seq_lens = [8]
        if cfg.max_seq_len not in warmup_seq_lens:
            warmup_seq_lens = list(warmup_seq_lens) + [cfg.max_seq_len]

        defer_capture = (
            cfg.use_cudagraph
            and cpp_state_dict is not None
            and kv_layers is None
        )
        (
            self.backbone,
            self.compile_seconds,
            self.vllm_config,
            self._cudagraph_input_pool,
            self._cudagraph_slot_pool,
        ) = compile_prefill_backbone(
            cfg,
            device=device,
            dtype=dtype,
            warmup_seq_lens=warmup_seq_lens,
            cpp_state_dict=cpp_state_dict,
            kv_layers=kv_layers,
            block_size=block_size,
            defer_cudagraph_capture=defer_capture,
        )
        self._cudagraph_replay_logged: set[str] = set()
        self._cudagraph_needs_reprime: set[int] = set()
        self._cudagraph_runtime_seen: set[int] = set()
        if cfg.use_cudagraph and kv_layers is not None and cpp_state_dict is not None:
            self._cudagraph_capture_done = True

    def ensure_cudagraph_capture(
        self,
        kv_layers,
        *,
        block_size: Optional[int] = None,
    ) -> None:
        """Finish deferred piecewise CUDAGraph capture once paged KV layers exist."""
        if not self.cfg.use_cudagraph or self._cudagraph_capture_done:
            return
        if self._cudagraph_input_pool is None:
            return
        bs = block_size if block_size is not None else self._block_size
        self._block_size = bs
        dev = next(self.backbone.inner.parameters()).device
        vocab = self.backbone.inner.config.vocab_size
        if self._cudagraph_slot_pool is None:
            self._cudagraph_slot_pool = _CudagraphSlotMappingPool(dev)
        capture_sizes = list(self.cfg.cudagraph_capture_sizes)
        _capture_cudagraphs(
            self.backbone,
            self.vllm_config,
            self.cfg,
            dev,
            vocab,
            capture_sizes,
            self._cudagraph_input_pool,
            kv_layers=kv_layers,
            slot_mapping_pool=self._cudagraph_slot_pool,
            block_size=bs,
            use_paged_ctx=True,
        )
        self._cudagraph_needs_reprime.clear()
        self._cudagraph_runtime_seen.clear()
        _snapshot_gpu_mem("T2_after_cudagraph_capture")
        self._cudagraph_capture_done = True

    @property
    def model(self):
        return self.backbone.inner

    def _maybe_log_replay_mem(self, bucket: int) -> None:
        for threshold, label in self._REPLAY_CHECKPOINTS:
            if bucket >= threshold and label not in self._cudagraph_replay_logged:
                _snapshot_gpu_mem(label, once=True)
                self._cudagraph_replay_logged.add(label)

    def _use_eager_prefill(self, input_ids: torch.Tensor) -> bool:
        return int(input_ids.shape[1]) < min_compiled_prefill_seq_len()

    def _compile_bucket_len(self, seq_len: int) -> int:
        cap = self.cfg.max_seq_len
        for bucket in compile_buckets(cap):
            if bucket > cap:
                continue
            if seq_len <= bucket:
                return bucket
        return cap

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        *,
        cudagraph_runtime_mode: Optional[object] = None,
    ) -> torch.Tensor:
        return _run_compiled_backbone(
            self.backbone,
            self.vllm_config,
            self.cfg,
            input_ids,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        )

    def _prepare_compiled_input_ids(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Pad to the Inductor warmup bucket so runtime stays on compiled bytecode."""
        seq_len = int(input_ids.shape[1])
        bucket = self._compile_bucket_len(seq_len)
        if bucket != seq_len:
            logger.info(
                "compiled prefill bucket: seq_len=%s bucket=%s pad=%s mode=%s",
                seq_len,
                bucket,
                max(0, bucket - seq_len),
                compile_bucket_mode(),
            )
        if self.cfg.use_cudagraph and self._cudagraph_input_pool is not None:
            staged = self._cudagraph_input_pool.stage(input_ids, bucket)
            self._maybe_log_replay_mem(bucket)
            return staged, seq_len
        if seq_len >= bucket:
            return input_ids, seq_len
        pad = torch.zeros(
            (input_ids.shape[0], bucket - seq_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, pad], dim=1), seq_len

    def _eager_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.backbone.inner.forward_prefill_compile(input_ids)
        return logits[0, -1, :]

    @torch.inference_mode()
    def run_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._use_eager_prefill(input_ids):
            return self.backbone.inner.forward_prefill_compile(input_ids)
        padded, seq_len = self._prepare_compiled_input_ids(input_ids)
        logits = self._run_backbone(padded)
        return logits[:, :seq_len, :]

    @torch.inference_mode()
    def run_prefill_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._use_eager_prefill(input_ids):
            return self._eager_last_logits(input_ids)
        padded, seq_len = self._prepare_compiled_input_ids(input_ids)
        logits = self._run_backbone(padded)
        return logits[0, seq_len - 1, :]

    @torch.inference_mode()
    def run_prefill_paged(
        self,
        input_ids: torch.Tensor,
        *,
        kv_layers,
        slot_mapping,
        block_size: int = 256,
    ) -> torch.Tensor:
        """Single compiled forward with graph-safe paged KV writes (shared C++ weights)."""
        from infinilm.torch_llama.kv_paged import PagedPrefillContext, paged_prefill_context
        from vllm.config import CUDAGraphMode

        seq_len = int(input_ids.shape[1])
        bucket = self._compile_bucket_len(seq_len)
        use_compiled = not self._use_eager_prefill(input_ids)
        staged_slots = slot_mapping
        # Bucket-padded slot pool is only valid on the compiled (padded) path.
        if (
            use_compiled
            and self.cfg.use_cudagraph
            and self._cudagraph_slot_pool is not None
        ):
            staged_slots = self._cudagraph_slot_pool.stage(
                slot_mapping, bucket, seq_len
            )
        ctx = PagedPrefillContext(
            kv_layers=list(kv_layers),
            slot_mapping=staged_slots,
            block_size=block_size,
        )
        if not use_compiled:
            with paged_prefill_context(ctx):
                logits = self.backbone.inner.forward_prefill_compile(input_ids)
            return logits[0, -1, :]

        with paged_prefill_context(ctx):
            padded, seq_len = self._prepare_compiled_input_ids(input_ids)
            cg_mode = cudagraph_runtime_mode_for_paged(self.cfg, seq_len, bucket)

            if cg_mode is CUDAGraphMode.PIECEWISE:
                prime_cudagraph_bucket_runtime(
                    cfg=self.cfg,
                    backbone=self.backbone,
                    vllm_config=self.vllm_config,
                    bucket=bucket,
                    kv_layers=kv_layers,
                    block_size=block_size,
                    input_pool=self._cudagraph_input_pool,
                    slot_pool=self._cudagraph_slot_pool,
                    needs_reprime=self._cudagraph_needs_reprime,
                    runtime_seen=self._cudagraph_runtime_seen,
                )
            if self.cfg.use_cudagraph and self._cudagraph_input_pool is not None:
                _maybe_log_cg_ptrs(
                    f"replay bucket={bucket} seq_len={seq_len} cg={cg_mode}",
                    input_ids=padded,
                    slot_mapping=self._cudagraph_slot_pool.get(bucket)
                    if self._cudagraph_slot_pool is not None
                    else None,
                )
            compiled_logits = self._run_backbone(padded, cudagraph_runtime_mode=cg_mode)

            if cg_mode is not None and cg_mode is not CUDAGraphMode.NONE:
                mark_partial_cudagraph_replay(
                    self.cfg, bucket, seq_len, self._cudagraph_needs_reprime
                )
            return compiled_logits[0, seq_len - 1, :]
