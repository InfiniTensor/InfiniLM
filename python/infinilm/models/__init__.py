"""
InfiniLM models
"""

from .llama import AutoLlamaModel,LlamaForCausalLM

__all__ = [
    "LlamaForCausalLM",
    "AutoLlamaModel"
]
