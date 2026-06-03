# Copyright (c) 2025, InfiniCore
"""vLLM-style torch.compile prefill backbone (dynamic seq 1..max_seq_len)."""

from .config import CompiledPrefillConfig
from .env import (
    compile_max_seq_len,
    prefill_compile_enabled,
    prefill_cudagraph_enabled,
    prefill_share_weights_enabled,
)
from .runner import CompiledPrefillRunner, min_compiled_prefill_seq_len

__all__ = [
    "CompiledPrefillConfig",
    "CompiledPrefillRunner",
    "compile_max_seq_len",
    "min_compiled_prefill_seq_len",
    "prefill_compile_enabled",
    "prefill_cudagraph_enabled",
    "prefill_share_weights_enabled",
]
