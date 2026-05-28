# Copyright (c) 2025, InfiniCore
"""Pure PyTorch Llama prefill backbone (flash-attn, LongRoPE) for compile / parity."""

from .model import TorchLlamaPrefillModel, load_torch_llama

__all__ = ["TorchLlamaPrefillModel", "load_torch_llama"]
