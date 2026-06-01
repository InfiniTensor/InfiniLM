# Copyright (c) 2025, InfiniCore
"""Init-compile torch prefill backbone with vLLM ``VllmBackend`` + ``CompilationConfig``."""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import torch

from infinilm.torch_llama.kv_paged import PagedPrefillContext, paged_prefill_context

from .backbone import VllmPrefillBackbone
from .config import CompiledPrefillConfig
from .env import compile_bucket_mode, compile_buckets
from .mem_profile import snapshot_gpu_mem

logger = logging.getLogger(__name__)

# Smallest init warmup length (``compile_warmup_seq_lens`` default). Shorter runtime
# lengths (chat template overhead) use eager forward.
_MIN_COMPILED_SEQ_LEN = 8


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
        use_cg = cfg.use_cudagraph and num_tokens in capture_set
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
) -> None:
    """Phase 2: piecewise CUDAGraph capture on already-compiled bytecode."""
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode

    # Large shapes first so smaller captures reuse the graph memory pool.
    sizes = sorted(capture_sizes, reverse=True)
    num_warmups = vllm_config.compilation_config.cudagraph_num_of_warmups
    logger.info(
        "compiled prefill: capturing piecewise CUDAGraphs for %s bucket(s)",
        len(sizes),
    )
    set_cudagraph_capturing_enabled(True)
    try:
        for seq_len in sizes:
            # Reuse persistent buffers so replay sees the same input addresses.
            dummy = input_pool.get(seq_len)
            dummy.zero_()
            with torch.inference_mode():
                # At least one NONE replay on the capture path before graph capture
                # (vLLM ``_capture_cudagraphs``; avoids legacy↔capture stream deps).
                none_runs = max(1, num_warmups)
                for _ in range(none_runs):
                    _run_compiled_backbone(
                        backbone,
                        vllm_config,
                        cfg,
                        dummy,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    )
                out = _run_compiled_backbone(
                    backbone,
                    vllm_config,
                    cfg,
                    dummy,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                )
            assert out.shape == (1, seq_len, vocab), out.shape
    finally:
        set_cudagraph_capturing_enabled(False)


def compile_prefill_backbone(
    cfg: CompiledPrefillConfig,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    warmup_seq_lens: Optional[List[int]] = None,
    cpp_state_dict: Optional[dict] = None,
) -> Tuple[VllmPrefillBackbone, float, object, Optional[_CudagraphInputPool]]:
    """
    Build ``VllmPrefillBackbone`` (vLLM torch.compile wrapper + ``VllmBackend``).

    Returns:
        (backbone, compile_seconds, vllm_config, input_pool)
        ``input_pool`` is set when ``cfg.use_cudagraph`` (persistent replay buffers).
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
        if cfg.use_cudagraph:
            input_pool = _CudagraphInputPool(dev)
            _warmup_compiled_backbone(
                backbone, vllm_config, cfg, dev, vocab, warmup_seq_lens
            )
            snapshot_gpu_mem("T1_after_inductor_warmup")
            _capture_cudagraphs(
                backbone,
                vllm_config,
                cfg,
                dev,
                vocab,
                list(cfg.cudagraph_capture_sizes),
                input_pool,
            )
            snapshot_gpu_mem("T2_after_cudagraph_capture")
        else:
            _warmup_compiled_backbone(
                backbone, vllm_config, cfg, dev, vocab, warmup_seq_lens
            )
            snapshot_gpu_mem("T1_after_inductor_warmup")

    elapsed = time.perf_counter() - t0
    vllm_config.compilation_config.compilation_time = elapsed
    logger.info("torch.compile takes %.3f s", elapsed)
    return backbone, elapsed, vllm_config, input_pool


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
    ):
        self.cfg = cfg
        if warmup_seq_lens is None:
            warmup_seq_lens = [8]
        if cfg.max_seq_len not in warmup_seq_lens:
            warmup_seq_lens = list(warmup_seq_lens) + [cfg.max_seq_len]

        (
            self.backbone,
            self.compile_seconds,
            self.vllm_config,
            self._cudagraph_input_pool,
        ) = compile_prefill_backbone(
            cfg,
            device=device,
            dtype=dtype,
            warmup_seq_lens=warmup_seq_lens,
            cpp_state_dict=cpp_state_dict,
        )
        self._cudagraph_replay_logged: set[str] = set()

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

    def _run_backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        return _run_compiled_backbone(
            self.backbone, self.vllm_config, self.cfg, input_ids
        )

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
        ctx = PagedPrefillContext(
            kv_layers=list(kv_layers),
            slot_mapping=slot_mapping,
            block_size=block_size,
        )
        if self._use_eager_prefill(input_ids):
            with paged_prefill_context(ctx):
                logits = self.backbone.inner.forward_prefill_compile(input_ids)
            return logits[0, -1, :]

        with paged_prefill_context(ctx):
            padded, seq_len = self._prepare_compiled_input_ids(input_ids)
            compiled_logits = self._run_backbone(padded)
            return compiled_logits[0, seq_len - 1, :]
