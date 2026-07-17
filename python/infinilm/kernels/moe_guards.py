# Copyright (c) 2025, InfiniCore
"""MoE serve guards + launcher hash (no Triton import)."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path


def launcher_source_path() -> Path:
    # Pinned launcher lives beside this module.
    return (Path(__file__).resolve().parent / "fused_moe_runtime.py").resolve()


def launcher_hash() -> str:
    return hashlib.sha256(launcher_source_path().read_bytes()).hexdigest()


def allow_moe_jit() -> bool:
    raw = os.environ.get("INFINI_MOE_ALLOW_JIT", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def assert_no_vllm(*, probe_import: bool = False) -> None:
    """Hard serve guard: vLLM must not be loaded alongside the pinned launcher.

    Recompile/warmup may set ``INFINI_MOE_ALLOW_JIT=1`` and use vLLM for parity;
    that path skips this guard. Serve (JIT off) always enforces it.
    """
    if allow_moe_jit():
        return
    # Hot-path: after first PASS, skip sys.modules scan (decode calls this every layer).
    cached = getattr(assert_no_vllm, "_ok", None)
    if cached is True and not probe_import:
        return
    if "vllm" in sys.modules or "vllm_mars" in sys.modules:
        raise RuntimeError(
            "fused_moe_runtime: vLLM is loaded (sys.modules). "
            "Serve must be vLLM-free; use offline rebuild/warmup for artifacts."
        )
    if probe_import:
        import importlib.util

        for name in ("vllm", "vllm_mars"):
            if importlib.util.find_spec(name) is not None:
                raise RuntimeError(
                    f"fused_moe_runtime: refusing serve with {name!r} importable. "
                    "Use a vLLM-free serve env; rebuild artifacts offline."
                )
    assert_no_vllm._ok = True  # type: ignore[attr-defined]
