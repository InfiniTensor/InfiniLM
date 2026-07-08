# Copyright (c) 2025, InfiniCore
"""PRD-04 unified torch.compile runner (plain torch.compile, no vLLM)."""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional

import torch
import torch._dynamo

from .config import TorchCompileConfig
from .env import (
    build_bs_to_padded_bucket,
    compile_buckets,
    compile_overflow_tail_bucket,
    padded_bucket_for_seq_len,
    torch_compile_share_weights_enabled,
)

logger = logging.getLogger(__name__)

__all__ = ["TorchCompileRunner"]

_VLLM_POWER_LADDER_CAP = 8192
_8192_EAGER_FALLBACK_LOGGED = False
_8192_PAD_WORKAROUND_LOGGED = False


class TorchCompileRunner:
    """Loads torch backbone, warms up per-bucket ``torch.compile``, runs prefill forward."""

    def __init__(
        self,
        cfg: TorchCompileConfig,
        *,
        device: Optional[torch.device] = None,
        cpp_state_dict: Optional[dict] = None,
        compile_mode: str = "default",
        mark_dynamic: bool = False,
    ):
        from infinilm.compile.env import check_torch_compile_mutual_exclusion

        check_torch_compile_mutual_exclusion()

        torch._dynamo.config.allow_unspec_int_on_nn_module = True

        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compile_mode = compile_mode
        self.mark_dynamic = mark_dynamic

        cfg.write_metadata()
        os.makedirs(cfg.cache_dir, exist_ok=True)
        inductor_cache = os.path.join(cfg.cache_dir, "inductor")
        os.makedirs(inductor_cache, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache

        if cpp_state_dict is None and torch_compile_share_weights_enabled():
            logger.warning(
                "INFINI_TORCH_COMPILE_SHARE_WEIGHTS=1 but cpp_state_dict is None; "
                "loading HF weights instead"
            )

        from infinilm.torch_llama.model import load_torch_llama

        self.model = load_torch_llama(
            cfg.model_path,
            device=self.device,
            splitting_flash_boundary=True,
            cpp_state_dict=cpp_state_dict,
        )

        compile_ladder = list(cfg.compile_sizes or compile_buckets(cfg.max_seq_len))
        self._bs_to_padded_bucket = build_bs_to_padded_bucket(compile_ladder)
        self._compile_bucket_fallback = (
            compile_ladder[-1] if compile_ladder else cfg.max_seq_len
        )
        self._overflow_tail_bucket = compile_overflow_tail_bucket(cfg.max_seq_len)
        self._compiled_fns: Dict[int, Callable] = {}
        self._eager_buckets: set[int] = set()
        pad_8192 = os.environ.get("INFINI_TORCH_COMPILE_8192_PAD", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        eager_env = os.environ.get("INFINI_TORCH_COMPILE_8192_EAGER")
        if eager_env is None:
            use_eager_8192 = not pad_8192
        else:
            use_eager_8192 = eager_env.strip().lower() in ("1", "true", "yes")
        if use_eager_8192:
            self._eager_buckets.add(_VLLM_POWER_LADDER_CAP)

    def _compile_bucket_len(self, seq_len: int) -> int:
        return padded_bucket_for_seq_len(
            seq_len,
            self._bs_to_padded_bucket,
            fallback=self._compile_bucket_fallback,
        )

    def _effective_compile_bucket(self, pad_bucket: int) -> int:
        """Pad 8192 compile bucket to overflow tail (8448) when pad workaround is enabled."""
        global _8192_PAD_WORKAROUND_LOGGED
        if pad_bucket in self._eager_buckets:
            return pad_bucket
        if (
            pad_bucket == _VLLM_POWER_LADDER_CAP
            and self._overflow_tail_bucket is not None
            and os.environ.get("INFINI_TORCH_COMPILE_8192_PAD", "0").strip().lower()
            in ("1", "true", "yes")
        ):
            if not _8192_PAD_WORKAROUND_LOGGED:
                logger.warning(
                    "8192 compile bucket: padding to overflow tail %s for compile "
                    "(set INFINI_TORCH_COMPILE_8192_EAGER=1 for eager fallback)",
                    self._overflow_tail_bucket,
                )
                _8192_PAD_WORKAROUND_LOGGED = True
            return self._overflow_tail_bucket
        return pad_bucket

    def _pad_input_ids(
        self, input_ids: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        seq_len = int(input_ids.shape[1])
        if seq_len >= target_len:
            return input_ids
        pad = torch.zeros(
            (input_ids.shape[0], target_len - seq_len),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, pad], dim=1)

    def _prepare_compiled_input_ids(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        """Pad to Inductor warmup bucket; return (padded_ids, seq_len, compile_bucket)."""
        seq_len = int(input_ids.shape[1])
        pad_bucket = self._compile_bucket_len(seq_len)
        compile_bucket = self._effective_compile_bucket(pad_bucket)
        padded = self._pad_input_ids(input_ids, pad_bucket)
        if compile_bucket > pad_bucket:
            padded = self._pad_input_ids(padded, compile_bucket)
        if pad_bucket != seq_len:
            logger.debug(
                "compiled prefill bucket: seq_len=%s pad_bucket=%s compile_bucket=%s",
                seq_len,
                pad_bucket,
                compile_bucket,
            )
        return padded, seq_len, compile_bucket

    def _get_compiled_fn(self, compile_bucket: int) -> Optional[Callable]:
        if compile_bucket in self._eager_buckets:
            global _8192_EAGER_FALLBACK_LOGGED
            if compile_bucket == _VLLM_POWER_LADDER_CAP and not _8192_EAGER_FALLBACK_LOGGED:
                logger.warning(
                    "8192 compile bucket: using eager forward (default or INFINI_TORCH_COMPILE_8192_EAGER=1)"
                )
                _8192_EAGER_FALLBACK_LOGGED = True
            return None
        if compile_bucket not in self._compiled_fns:
            forward = self.model.forward_prefill_compile
            compiled = torch.compile(
                forward,
                mode=self.compile_mode,
                fullgraph=False,
            )
            self._compiled_fns[compile_bucket] = compiled
        return self._compiled_fns[compile_bucket]

    def _maybe_mark_dynamic(self, input_ids: torch.Tensor) -> None:
        if self.mark_dynamic:
            torch._dynamo.mark_dynamic(input_ids, 1)

    def warmup(self) -> None:
        """Compile and run each bucket in ``cfg.compile_sizes``."""
        for bucket in self.cfg.compile_sizes or ():
            bucket = int(bucket)
            if (
                self._overflow_tail_bucket is not None
                and bucket == self._overflow_tail_bucket
                and _VLLM_POWER_LADDER_CAP in self._eager_buckets
            ):
                continue
            if (
                self._overflow_tail_bucket is not None
                and bucket == self._overflow_tail_bucket
                and bucket != self._effective_compile_bucket(_VLLM_POWER_LADDER_CAP)
            ):
                logger.info(
                    "deferring overflow-tail bucket %s warmup (compile on first 8192 request)",
                    bucket,
                )
                continue
            compile_bucket = self._effective_compile_bucket(bucket)
            if bucket in self._eager_buckets:
                logger.info("skipping warmup for eager bucket %s", bucket)
                continue
            compiled_fn = self._get_compiled_fn(compile_bucket)
            if compiled_fn is None:
                continue
            dummy = torch.zeros(
                (1, compile_bucket), dtype=torch.long, device=self.device
            )
            _ = compiled_fn(dummy)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run_prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full logits ``[batch, seq_len, vocab]`` (unpadded)."""
        padded, seq_len, compile_bucket = self._prepare_compiled_input_ids(input_ids)
        compiled_fn = self._get_compiled_fn(compile_bucket)
        from infinilm.torch_llama.prefill_context import prefill_compile_context

        with prefill_compile_context(seq_len):
            if compiled_fn is None:
                logits = self.model.forward_prefill_compile(
                    padded, valid_seq_len=seq_len
                )
            else:
                self._maybe_mark_dynamic(padded)
                logits = compiled_fn(padded)
        return logits[:, :seq_len, :]

    @torch.inference_mode()
    def run_prefill_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Last valid position logits ``[batch, vocab]``."""
        padded, seq_len, compile_bucket = self._prepare_compiled_input_ids(input_ids)
        compiled_fn = self._get_compiled_fn(compile_bucket)
        from infinilm.torch_llama.prefill_context import prefill_compile_context

        with prefill_compile_context(seq_len):
            if compiled_fn is None:
                logits = self.model.forward_prefill_compile(
                    padded, valid_seq_len=seq_len
                )
            else:
                self._maybe_mark_dynamic(padded)
                logits = compiled_fn(padded)
        return logits[0, seq_len - 1, :]

    @torch.inference_mode()
    def run_decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("decode compile path is M4")
