# Copyright (c) 2025, InfiniCore
"""Pure PyTorch Llama prefill backbone (flash-attn, LongRoPE) for compile / parity."""

from __future__ import annotations

from typing import Any

__all__ = ["TorchLlamaPrefillModel", "load_torch_llama"]


def __getattr__(name: str) -> Any:
    if name in ("TorchLlamaPrefillModel", "load_torch_llama"):
        from .model import TorchLlamaPrefillModel, load_torch_llama

        return TorchLlamaPrefillModel if name == "TorchLlamaPrefillModel" else load_torch_llama
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
