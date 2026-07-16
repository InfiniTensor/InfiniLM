"""InfiniLM public package (lazy exports so tools/submodules stay light)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AutoLlamaModel",
    "distributed",
    "cache",
    "llm",
    "base_config",
    "LLM",
    "AsyncLLMEngine",
    "SamplingParams",
    "RequestOutput",
    "TokenOutput",
]


def __getattr__(name: str) -> Any:
    if name == "AutoLlamaModel":
        from .models import AutoLlamaModel

        return AutoLlamaModel
    if name in ("distributed", "cache", "llm", "base_config"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    if name in (
        "LLM",
        "AsyncLLMEngine",
        "SamplingParams",
        "RequestOutput",
        "TokenOutput",
    ):
        from . import llm as _llm

        return getattr(_llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
