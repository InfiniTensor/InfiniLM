# Copyright (c) 2025, InfiniCore
"""Init-compile torch prefill backbone with vLLM ``VllmBackend`` + ``CompilationConfig``."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .cudagraph_capture import _capture_cudagraphs, _warmup_compiled_backbone, recapture_cudagraph_buckets, refresh_cudagraph_wrapper_refs, sync_captured_buckets_from_wrappers
from .cudagraph_pools import (
    _CudagraphInputPool,
    _CudagraphKvStagingPool,
    _CudagraphSlotMappingPool,
    _CudagraphValidLenPool,
    _maybe_log_cg_ptrs,
    kv_staging_context,
    valid_seq_len_context,
)
from .cudagraph_runtime import (
    buckets_needing_recapture,
    conservative_reprime_before_piecewise,
    cudagraph_runtime_mode_for_paged,
    mark_partial_cudagraph_replay,
    min_compiled_prefill_seq_len,
    prime_cudagraph_bucket_runtime,
    _run_compiled_backbone,
)
from .env import (
    compile_bucket_mode,
    compile_buckets,
    cudagraph_pool_tier_id,
    prefill_cg_kv_outside_graph,
    prefill_cg_pool_tier_isolation,
    prefill_cg_valid_seq_len,
)

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


def _make_valid_len_pool(
    device: torch.device,
    *,
    capture_buckets: Optional[List[int]] = None,
) -> Optional[_CudagraphValidLenPool]:
    if not prefill_cg_valid_seq_len():
        return None
    return _CudagraphValidLenPool(device, capture_buckets=capture_buckets)


def _make_kv_staging_pool(
    backbone: VllmPrefillBackbone,
    device: torch.device,
    *,
    capture_buckets: Optional[List[int]] = None,
) -> Optional[_CudagraphKvStagingPool]:
    if not prefill_cg_kv_outside_graph():
        return None
    cfg = backbone.inner.config
    num_layers = int(getattr(cfg, "num_hidden_layers", 0))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", 0))
    head_dim = int(getattr(cfg, "head_dim", 0))
    if head_dim <= 0 and num_kv_heads > 0:
        head_dim = int(getattr(cfg, "hidden_size", 0)) // max(
            int(getattr(cfg, "num_attention_heads", 1)), 1
        )
    dtype = next(backbone.inner.parameters()).dtype
    if num_layers <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError(
            f"invalid KV staging dims: layers={num_layers} "
            f"kv_heads={num_kv_heads} head_dim={head_dim}"
        )
    return _CudagraphKvStagingPool(
        device,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        capture_buckets=capture_buckets,
    )


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
    Optional[_CudagraphKvStagingPool],
    Optional[_CudagraphValidLenPool],
]:
    """
    Build ``VllmPrefillBackbone`` (vLLM torch.compile wrapper + ``VllmBackend``).

    Returns:
        (backbone, compile_seconds, vllm_config, input_pool, slot_pool)
        ``input_pool`` / ``slot_mapping_pool`` when ``cfg.use_cudagraph``.
        ``kv_staging_pool`` when ``INFINI_PREFILL_CG_KV_OUTSIDE_GRAPH=1``.
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
        kv_staging_pool: Optional[_CudagraphKvStagingPool] = None
        valid_len_pool: Optional[_CudagraphValidLenPool] = None
        use_paged_capture = cpp_state_dict is not None
        kv_outside_graph = prefill_cg_kv_outside_graph()
        if cfg.use_cudagraph:
            input_pool = _CudagraphInputPool(dev)
            if use_paged_capture:
                slot_pool = _CudagraphSlotMappingPool(dev)
            capture_bucket_list = list(cfg.cudagraph_capture_sizes or ())
            kv_staging_pool = _make_kv_staging_pool(
                backbone,
                dev,
                capture_buckets=capture_bucket_list,
            )
            valid_len_pool = _make_valid_len_pool(
                dev,
                capture_buckets=capture_bucket_list,
            )
            _warmup_compiled_backbone(
                backbone, vllm_config, cfg, dev, vocab, warmup_seq_lens,
                kv_staging_pool=kv_staging_pool,
                valid_len_pool=valid_len_pool,
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
                    kv_staging_pool=kv_staging_pool,
                    valid_len_pool=valid_len_pool,
                    block_size=block_size,
                    use_paged_ctx=use_paged_capture
                    and kv_layers is not None
                    and not kv_outside_graph,
                )
                refresh_cudagraph_wrapper_refs(backbone)
                _snapshot_gpu_mem("T2_after_cudagraph_capture")
            elif defer_cudagraph_capture:
                logger.info(
                    "compiled prefill: deferring CUDAGraph capture until kv_layers "
                    "are available (share_weights + paged KV path)"
                )
        else:
            _warmup_compiled_backbone(
                backbone, vllm_config, cfg, dev, vocab, warmup_seq_lens,
                kv_staging_pool=kv_staging_pool,
                valid_len_pool=valid_len_pool,
            )
            _snapshot_gpu_mem("T1_after_inductor_warmup")

    elapsed = time.perf_counter() - t0
    vllm_config.compilation_config.compilation_time = elapsed
    logger.info("torch.compile takes %.3f s", elapsed)
    return backbone, elapsed, vllm_config, input_pool, slot_pool, kv_staging_pool, valid_len_pool


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
            self._cudagraph_kv_staging_pool,
            self._cudagraph_valid_len_pool,
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
        self._cudagraph_captured_buckets: set[int] = set()
        self._cudagraph_last_forward_bucket: Optional[int] = None
        self._cudagraph_wrappers: list = []
        if cfg.use_cudagraph and kv_layers is not None and cpp_state_dict is not None:
            self._cudagraph_capture_done = True
        if self._cudagraph_capture_done:
            for b in self.cfg.cudagraph_capture_sizes or ():
                self._cudagraph_runtime_seen.add(int(b))
            refresh_cudagraph_wrapper_refs(self.backbone)
            self._cudagraph_wrappers = list(
                getattr(self.backbone, "_cudagraph_wrappers", [])
            )
            self._cudagraph_captured_buckets = sync_captured_buckets_from_wrappers(
                self.backbone, self.cfg.cudagraph_capture_sizes
            )

    def _sync_captured_buckets(self) -> None:
        self._cudagraph_captured_buckets = sync_captured_buckets_from_wrappers(
            self.backbone, self.cfg.cudagraph_capture_sizes
        )

    def recapture_cudagraph_for_buckets(
        self,
        buckets: List[int],
        *,
        kv_layers,
        block_size: Optional[int] = None,
    ) -> None:
        """Force piecewise re-capture (parity gates / explicit poison recovery)."""
        if not self.cfg.use_cudagraph or self._cudagraph_input_pool is None:
            return
        bs = block_size if block_size is not None else self._block_size
        unique = sorted({int(b) for b in buckets if b})
        if not unique:
            return
        kv_outside = prefill_cg_kv_outside_graph() and self._cudagraph_kv_staging_pool is not None
        dev = next(self.backbone.inner.parameters()).device
        vocab = self.backbone.inner.config.vocab_size
        recapture_cudagraph_buckets(
            self.backbone,
            self.vllm_config,
            self.cfg,
            dev,
            vocab,
            unique,
            self._cudagraph_input_pool,
            kv_layers=kv_layers,
            slot_mapping_pool=self._cudagraph_slot_pool,
            kv_staging_pool=self._cudagraph_kv_staging_pool,
            valid_len_pool=self._cudagraph_valid_len_pool,
            block_size=bs,
            use_paged_ctx=not kv_outside,
            capture_order="descending",
        )
        for b in unique:
            self._cudagraph_needs_reprime.discard(b)
            self._cudagraph_runtime_seen.add(b)
        self._cudagraph_last_forward_bucket = None
        refresh_cudagraph_wrapper_refs(self.backbone)
        self._cudagraph_wrappers = list(
            getattr(self.backbone, "_cudagraph_wrappers", [])
        )
        self._sync_captured_buckets()

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
            kv_staging_pool=self._cudagraph_kv_staging_pool,
            valid_len_pool=self._cudagraph_valid_len_pool,
            block_size=bs,
            use_paged_ctx=not prefill_cg_kv_outside_graph(),
        )
        self._cudagraph_needs_reprime.clear()
        self._cudagraph_runtime_seen.clear()
        self._cudagraph_last_forward_bucket = None
        for b in capture_sizes:
            self._cudagraph_runtime_seen.add(int(b))
        refresh_cudagraph_wrapper_refs(self.backbone)
        self._cudagraph_wrappers = list(self.backbone._cudagraph_wrappers)
        self._sync_captured_buckets()
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
        seq_len: Optional[int] = None,
        bucket: Optional[int] = None,
    ) -> torch.Tensor:
        cm: contextlib.AbstractContextManager = contextlib.nullcontext()
        if (
            self._cudagraph_valid_len_pool is not None
            and bucket is not None
            and seq_len is not None
        ):
            valid_tensor, valid_mask = self._cudagraph_valid_len_pool.stage(
                seq_len, bucket
            )
            cm = valid_seq_len_context(valid_tensor, mask=valid_mask)
        with cm:
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
        from infinilm.torch_llama.kv_paged import (
            PagedPrefillContext,
            flush_staged_kv_to_paged_cache,
            paged_prefill_context,
        )
        from vllm.config import CUDAGraphMode

        seq_len = int(input_ids.shape[1])
        bucket = self._compile_bucket_len(seq_len)
        use_compiled = not self._use_eager_prefill(input_ids)
        kv_outside = prefill_cg_kv_outside_graph() and self._cudagraph_kv_staging_pool is not None
        staged_slots = slot_mapping
        if self.cfg.use_cudagraph and self._cudagraph_slot_pool is not None:
            stage_bucket = bucket if use_compiled else seq_len
            staged_slots = self._cudagraph_slot_pool.stage(
                slot_mapping, stage_bucket, seq_len
            )
        flush_ctx = PagedPrefillContext(
            kv_layers=list(kv_layers),
            slot_mapping=staged_slots,
            block_size=block_size,
        )
        if not use_compiled:
            with paged_prefill_context(flush_ctx):
                logits = self.backbone.inner.forward_prefill_compile(input_ids)
            return logits[0, -1, :]

        padded, seq_len = self._prepare_compiled_input_ids(input_ids)
        cg_mode = cudagraph_runtime_mode_for_paged(
            self.cfg,
            seq_len,
            bucket,
            captured_buckets=self._cudagraph_captured_buckets,
        )

        if (
            self.cfg.use_cudagraph
            and prefill_cg_pool_tier_isolation()
            and self._cudagraph_last_forward_bucket is not None
        ):
            cur_tier = cudagraph_pool_tier_id(bucket)
            last_tier = cudagraph_pool_tier_id(self._cudagraph_last_forward_bucket)
            if cur_tier != last_tier:
                self._cudagraph_needs_reprime = {
                    b
                    for b in self._cudagraph_needs_reprime
                    if cudagraph_pool_tier_id(b) == cur_tier
                }

        if cg_mode is CUDAGraphMode.PIECEWISE:
            conservative_reprime_before_piecewise(
                self.cfg,
                bucket,
                self._cudagraph_last_forward_bucket,
                self._cudagraph_needs_reprime,
            )

        recaptured: set[int] = set()
        poisoned = buckets_needing_recapture(
            self._cudagraph_needs_reprime,
            bucket,
            self.cfg.cudagraph_capture_sizes,
        )
        if self.cfg.use_cudagraph and poisoned and self._cudagraph_input_pool is not None:
            logger.info(
                "compiled prefill: poisoned ladder before bucket %s; "
                "re-capturing %s",
                bucket,
                poisoned,
            )
            dev = next(self.backbone.inner.parameters()).device
            vocab = self.backbone.inner.config.vocab_size
            recapture_cudagraph_buckets(
                self.backbone,
                self.vllm_config,
                self.cfg,
                dev,
                vocab,
                poisoned,
                self._cudagraph_input_pool,
                kv_layers=kv_layers,
                slot_mapping_pool=self._cudagraph_slot_pool,
                kv_staging_pool=self._cudagraph_kv_staging_pool,
                valid_len_pool=self._cudagraph_valid_len_pool,
                block_size=block_size,
                use_paged_ctx=not kv_outside,
                capture_order="descending",
            )
            for b in poisoned:
                self._cudagraph_needs_reprime.discard(b)
                self._cudagraph_runtime_seen.add(b)
                recaptured.add(b)
            refresh_cudagraph_wrapper_refs(self.backbone)
            self._cudagraph_wrappers = list(
                getattr(self.backbone, "_cudagraph_wrappers", [])
            )
            self._sync_captured_buckets()

        if cg_mode is CUDAGraphMode.PIECEWISE and bucket not in recaptured:
            prime_cudagraph_bucket_runtime(
                cfg=self.cfg,
                backbone=self.backbone,
                vllm_config=self.vllm_config,
                bucket=bucket,
                kv_layers=kv_layers,
                block_size=block_size,
                input_pool=self._cudagraph_input_pool,
                slot_pool=self._cudagraph_slot_pool,
                kv_staging_pool=self._cudagraph_kv_staging_pool,
                valid_len_pool=self._cudagraph_valid_len_pool,
                needs_reprime=self._cudagraph_needs_reprime,
                runtime_seen=self._cudagraph_runtime_seen,
                prime_seq_len=seq_len if seq_len < bucket else None,
                reuse_staged_input=seq_len < bucket,
            )
        if self.cfg.use_cudagraph and self._cudagraph_input_pool is not None:
            _maybe_log_cg_ptrs(
                f"replay bucket={bucket} seq_len={seq_len} cg={cg_mode}",
                input_ids=padded,
                slot_mapping=self._cudagraph_slot_pool.get(bucket)
                if self._cudagraph_slot_pool is not None
                else None,
            )

        if kv_outside:
            with kv_staging_context(self._cudagraph_kv_staging_pool, bucket):
                compiled_logits = self._run_backbone(
                    padded,
                    cudagraph_runtime_mode=cg_mode,
                    seq_len=seq_len,
                    bucket=bucket,
                )
            flush_staged_kv_to_paged_cache(
                self._cudagraph_kv_staging_pool,
                bucket,
                seq_len,
                flush_ctx,
            )
        else:
            with paged_prefill_context(flush_ctx):
                compiled_logits = self._run_backbone(
                    padded,
                    cudagraph_runtime_mode=cg_mode,
                    seq_len=seq_len,
                    bucket=bucket,
                )

        if self.cfg.use_cudagraph and seq_len < bucket:
            mark_partial_cudagraph_replay(
                self.cfg,
                bucket,
                seq_len,
                self._cudagraph_needs_reprime,
                cudagraph_runtime_mode=cg_mode,
            )
        self._cudagraph_last_forward_bucket = bucket
        return compiled_logits[0, seq_len - 1, :]
