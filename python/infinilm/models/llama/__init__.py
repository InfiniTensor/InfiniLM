"""
InfiniLM Llama interfaces.

This package exposes two flavors of the Llama stack:

1. Native InfiniCore-backed bindings implemented in C++ (exported as
   ``LlamaConfig``, ``LlamaModel``, ``LlamaForCausalLM`` and ``Device``).
2. Transformer-compatible Python reference implementations derived from
   HuggingFace (exported under the ``Transformers*`` aliases).
"""

from . import configuration_llama
from . import modeling_llama
from .llama_cpp import LlamaForCausalLM

# Provide explicit aliases for the Transformers-style Python implementation so
# downstream tooling can still reach it without clashing with the native API.
TransformersLlamaConfig = configuration_llama.LlamaConfig
TransformersLlamaModel = modeling_llama.LlamaModel
TransformersLlamaForCausalLM = modeling_llama.LlamaForCausalLM

__all__ = [
    # Native C++ bindings
    "LlamaForCausalLM",
    # Transformer-style reference implementation
    "configuration_llama",
    "modeling_llama",
    "TransformersLlamaConfig",
    "TransformersLlamaModel",
    "TransformersLlamaForCausalLM",
]
