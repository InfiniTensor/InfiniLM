# Copyright (c) 2025, InfiniCore
"""vLLM-style torch.compile prefill backbone (dynamic seq 1..max_seq_len)."""

from .config import (
    DEFAULT_SPLITTING_OPS,
    CompiledPrefillConfig,
    default_compile_size_ladder,
)
from .env import (
    compile_max_seq_len,
    prefill_compile_enabled,
    prefill_cudagraph_enabled,
    prefill_share_weights_enabled,
)
from .runner import CompiledPrefillRunner, compile_prefill_backbone
from .weights import bind_cpp_weights_to_torch

__all__ = [
    "DEFAULT_SPLITTING_OPS",
    "CompiledPrefillConfig",
    "CompiledPrefillRunner",
    "bind_cpp_weights_to_torch",
    "compile_max_seq_len",
    "compile_prefill_backbone",
    "default_compile_size_ladder",
    "prefill_compile_enabled",
    "prefill_cudagraph_enabled",
    "prefill_share_weights_enabled",
]
