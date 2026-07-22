# Copyright (c) 2025, InfiniCore
"""Serve-time MiniCPM5 MoE artifact validation (no InfiniCore native deps)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def assert_moe_compile_on_miss_disabled() -> None:
    """Raise if MoE serve attempts compile-on-miss (no escape hatch)."""
    com_raw = os.environ.get("INFINI_PIECEWISE_INDUCTOR_COMPILE_ON_MISS")
    if com_raw is not None and com_raw.strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise RuntimeError(
            "minicpm5_moe forbids INFINI_PIECEWISE_INDUCTOR_COMPILE_ON_MISS=1 "
            "(compile-on-miss is deprecated for all models). "
            "Offline AOT: python -m infinilm.server.entry --phase compile|all "
            "or ./scripts/rebuild_minicpm5_moe_artifacts.sh"
        )
    os.environ["INFINI_PIECEWISE_INDUCTOR_COMPILE_ON_MISS"] = "0"


def validate_minicpm5_moe_artifacts(*, model_path: str, cache_root: str) -> dict[str, Any]:
    """Mandatory MoE artifact checks (no optional strict flag). Raises on failure."""
    from infinilm.compile.config import model_cache_hash
    from infinilm.kernels.moe_guards import assert_no_vllm, launcher_hash
    from infinilm.tools.pack_moe_artifacts import DEFAULT_BUCKETS, verify_bundle

    model_dir = Path(cache_root) / model_cache_hash(model_path)
    configs = model_dir / "moe_configs"
    triton_cache = model_dir / "moe_triton_cache"
    if not os.environ.get("INFINI_MOE_CONFIGS", "").strip():
        os.environ["INFINI_MOE_CONFIGS"] = str(configs)
    if not os.environ.get("INFINI_MOE_TRITON_CACHE", "").strip():
        os.environ["INFINI_MOE_TRITON_CACHE"] = str(triton_cache)
    os.environ.setdefault("TRITON_CACHE_DIR", os.environ["INFINI_MOE_TRITON_CACHE"])
    jit_raw = os.environ.get("INFINI_MOE_ALLOW_JIT", "0").strip().lower()
    if jit_raw in ("1", "true", "yes", "on"):
        raise RuntimeError(
            "minicpm5_moe serve forbids INFINI_MOE_ALLOW_JIT=1. "
            "Unset it and serve from a pre-warmed moe_triton_cache."
        )
    os.environ["INFINI_MOE_ALLOW_JIT"] = "0"
    assert_no_vllm()

    info = verify_bundle(
        cache_root=Path(cache_root),
        model_path=model_path,
        buckets=list(DEFAULT_BUCKETS),
    )
    env_configs = Path(os.environ["INFINI_MOE_CONFIGS"]).resolve()
    env_cache = Path(os.environ["INFINI_MOE_TRITON_CACHE"]).resolve()
    if env_configs != configs.resolve():
        raise RuntimeError(
            f"INFINI_MOE_CONFIGS={env_configs} does not match bundle {configs}"
        )
    if env_cache != triton_cache.resolve():
        raise RuntimeError(
            f"INFINI_MOE_TRITON_CACHE={env_cache} does not match bundle {triton_cache}"
        )
    logger.info(
        "minicpm5_moe artifacts OK: manifest=%s launcher_hash=%s buckets=%s",
        info.get("manifest"),
        launcher_hash(),
        info.get("buckets"),
    )
    return info
