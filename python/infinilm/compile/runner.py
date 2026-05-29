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

logger = logging.getLogger(__name__)

# Smallest init warmup length (``compile_warmup_seq_lens`` default). Shorter runtime
# lengths (chat template overhead) use eager forward.
_MIN_COMPILED_SEQ_LEN = 8
def _compile_buckets(max_seq_len: int) -> Tuple[int, ...]:
    """Coarse buckets aligned with default ``COMPILE_WARMUP_SEQ_LENS`` / Inductor ladder."""
    buckets: List[int] = [512, 1024, 4096]
    if max_seq_len >= 8192:
        buckets.append(8192)
    if max_seq_len >= 8448:
        buckets.append(8448)
    elif max_seq_len not in buckets:
        buckets.append(max_seq_len)
    return tuple(buckets)


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
) -> Tuple[VllmPrefillBackbone, float, object]:
    """
    Build ``VllmPrefillBackbone`` (vLLM torch.compile wrapper + ``VllmBackend``).

    Returns:
        (backbone, compile_seconds, vllm_config)
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
        for seq_len in warmup_seq_lens:
            dummy = torch.zeros((1, seq_len), dtype=torch.long, device=dev)
            with torch.inference_mode():
                out = backbone(dummy)
            assert out.shape == (1, seq_len, vocab), out.shape

    elapsed = time.perf_counter() - t0
    vllm_config.compilation_config.compilation_time = elapsed
    logger.info("torch.compile takes %.3f s", elapsed)
    return backbone, elapsed, vllm_config


class CompiledPrefillRunner:
    """Loads torch backbone, runs init compile, exposes compiled prefill forward."""

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

        self.backbone, self.compile_seconds, self.vllm_config = compile_prefill_backbone(
            cfg,
            device=device,
            dtype=dtype,
            warmup_seq_lens=warmup_seq_lens,
            cpp_state_dict=cpp_state_dict,
        )

    @property
    def model(self):
        return self.backbone.inner

    def _use_eager_prefill(self, input_ids: torch.Tensor) -> bool:
        return int(input_ids.shape[1]) < _MIN_COMPILED_SEQ_LEN

    def _compile_bucket_len(self, seq_len: int) -> int:
        cap = self.cfg.max_seq_len
        for bucket in _compile_buckets(cap):
            if bucket > cap:
                continue
            if seq_len <= bucket:
                return bucket
        return cap

    def _eager_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.backbone.inner.forward_prefill_compile(input_ids)
        return logits[0, -1, :]

    def _prepare_compiled_input_ids(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Pad to the Inductor warmup bucket so runtime stays on compiled bytecode."""
        seq_len = int(input_ids.shape[1])
        bucket = self._compile_bucket_len(seq_len)
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
        logits = self.backbone(padded)
        return logits[:, :seq_len, :]

    @torch.inference_mode()
    def run_prefill_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._use_eager_prefill(input_ids):
            return self._eager_last_logits(input_ids)
        padded, seq_len = self._prepare_compiled_input_ids(input_ids)
        logits = self.backbone(padded)
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
            compiled_logits = self.backbone(padded)
            return compiled_logits[0, seq_len - 1, :]
