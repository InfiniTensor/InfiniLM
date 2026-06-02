# Copyright (c) 2025, InfiniCore
"""Init-compile torch prefill backbone with vLLM ``VllmBackend`` + ``CompilationConfig``."""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple, Union

import infinicore
import torch

from infinilm.torch_llama.kv_paged import PagedPrefillContext, paged_prefill_context

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .env import compile_bucket_mode, compile_buckets, prefill_cg_debug_ptrs_enabled
from .mem_profile import snapshot_gpu_mem

logger = logging.getLogger(__name__)

# Smallest init warmup length (``compile_warmup_seq_lens`` default). Shorter runtime
# lengths (chat template overhead) use eager forward.
_MIN_COMPILED_SEQ_LEN = 8


def min_compiled_prefill_seq_len() -> int:
    """Public accessor for server hybrid gating."""
    return _MIN_COMPILED_SEQ_LEN


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


def _paged_capture_effective_seq_len(bucket: int, max_seq: int) -> int:
    """Valid slot prefix length at CUDAGraph capture (must be >= any replay ``seq_len`` in bucket).

    Capture must upper-bound replay paged-write footprint: all bucket positions get
    real slot indices; ``slot_mapping_pool.synthetic`` still pads ``buf[seq_len:bucket]``
    with ``-1`` at replay when ``seq_len < bucket``.
    """
    del max_seq  # ladder is implicit in ``bucket``; kept for call-site stability.
    return bucket


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
            and num_tokens >= _PARTIAL_CG_PIECEWISE_MIN_BUCKET
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


# Partial PIECEWISE replay below this bucket poisons larger CUDAGraphs on MetaX.
_PARTIAL_CG_PIECEWISE_MIN_BUCKET = 4096


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
        snapshot_gpu_mem("T0_model_load")
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
            snapshot_gpu_mem("T1_after_inductor_warmup")
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
                snapshot_gpu_mem("T2_after_cudagraph_capture")
            elif defer_cudagraph_capture:
                logger.info(
                    "compiled prefill: deferring CUDAGraph capture until kv_layers "
                    "are available (share_weights + paged KV path)"
                )
        else:
            _warmup_compiled_backbone(
                backbone, vllm_config, cfg, dev, vocab, warmup_seq_lens
            )
            snapshot_gpu_mem("T1_after_inductor_warmup")

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
        snapshot_gpu_mem("T2_after_cudagraph_capture")
        self._cudagraph_capture_done = True

    @property
    def model(self):
        return self.backbone.inner

    def _maybe_log_replay_mem(self, bucket: int) -> None:
        for threshold, label in self._REPLAY_CHECKPOINTS:
            if bucket >= threshold and label not in self._cudagraph_replay_logged:
                snapshot_gpu_mem(label, once=True)
                self._cudagraph_replay_logged.add(label)

    def _use_eager_prefill(self, input_ids: torch.Tensor) -> bool:
        return int(input_ids.shape[1]) < _MIN_COMPILED_SEQ_LEN

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

    def _mark_partial_cudagraph_replay(self, bucket: int, seq_len: int) -> None:
        if seq_len >= bucket or not self.cfg.use_cudagraph:
            return
        capture_set = set(self.cfg.cudagraph_capture_sizes or ())
        for b in capture_set:
            if b > bucket:
                self._cudagraph_needs_reprime.add(b)

    def _cudagraph_runtime_mode_for_paged(
        self, seq_len: int, bucket: int
    ) -> Optional[object]:
        """CUDAGraph replay mode for bucket-padded compiled prefill.

        * ``seq_len < bucket``: always ``NONE`` (no partial PIECEWISE; MetaX ATU).
        * ``seq_len == bucket`` and bucket ≥ 4096: ``PIECEWISE`` full-bucket replay.
        * Repro: ``INFINI_PREFILL_CG_ALLOW_PARTIAL=1`` restores auto PIECEWISE on partial.
        """
        from .env import prefill_cg_allow_partial_pad
        from vllm.config import CUDAGraphMode

        if not self.cfg.use_cudagraph:
            return None
        if prefill_cg_allow_partial_pad():
            return None
        if seq_len < bucket:
            return CUDAGraphMode.NONE
        if seq_len == bucket and bucket >= _PARTIAL_CG_PIECEWISE_MIN_BUCKET:
            return CUDAGraphMode.PIECEWISE
        return CUDAGraphMode.NONE

    def _prime_cudagraph_bucket_runtime(
        self,
        bucket: int,
        *,
        kv_layers,
        block_size: int,
    ) -> bool:
        if (
            not self.cfg.use_cudagraph
            or self._cudagraph_input_pool is None
            or self._cudagraph_slot_pool is None
        ):
            return False
        if bucket in self._cudagraph_runtime_seen and bucket not in self._cudagraph_needs_reprime:
            return False
        if (
            bucket < _PARTIAL_CG_PIECEWISE_MIN_BUCKET
            and bucket not in self._cudagraph_needs_reprime
        ):
            return False
        from vllm.config import CUDAGraphMode

        dummy = self._cudagraph_input_pool.get(bucket)
        dummy.zero_()
        staged_slots = self._cudagraph_slot_pool.synthetic(
            bucket,
            effective_seq_len=_paged_capture_effective_seq_len(
                bucket, self.cfg.max_seq_len
            ),
        )
        ctx = PagedPrefillContext(
            kv_layers=list(kv_layers),
            slot_mapping=staged_slots,
            block_size=block_size,
        )
        with paged_prefill_context(ctx):
            self._run_backbone(
                dummy,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            )
        self._cudagraph_needs_reprime.discard(bucket)
        self._cudagraph_runtime_seen.add(bucket)
        return True

    def _eager_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.backbone.inner.forward_prefill_compile(input_ids)
        return logits[0, -1, :]

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
            cg_mode = self._cudagraph_runtime_mode_for_paged(seq_len, bucket)
            from vllm.config import CUDAGraphMode

            if cg_mode is CUDAGraphMode.PIECEWISE:
                self._prime_cudagraph_bucket_runtime(
                    bucket, kv_layers=kv_layers, block_size=block_size
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
                self._mark_partial_cudagraph_replay(bucket, seq_len)
            return compiled_logits[0, seq_len - 1, :]
