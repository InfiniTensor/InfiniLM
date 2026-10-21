"""Minimal CUDA-graph policy env expand for full_and_piecewise."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

CUDAGRAPH_POLICY_EAGER = "eager"
CUDAGRAPH_POLICY_FULL_AND_PIECEWISE = "full_and_piecewise"


def cudagraph_policy() -> str:
    raw = os.environ.get("INFINI_CUDAGRAPH_POLICY", "").strip().lower()
    if raw in ("", CUDAGRAPH_POLICY_EAGER):
        return CUDAGRAPH_POLICY_EAGER if raw == CUDAGRAPH_POLICY_EAGER else ""
    if raw == CUDAGRAPH_POLICY_FULL_AND_PIECEWISE:
        return CUDAGRAPH_POLICY_FULL_AND_PIECEWISE
    return ""


def apply_cudagraph_policy_env(policy: Optional[str] = None) -> str:
    """When F&P and FA_INGRAPH unset → decode (matches C++ faIngraphMode_)."""
    pol = (policy or os.environ.get("INFINI_CUDAGRAPH_POLICY", "")).strip().lower()
    if pol == CUDAGRAPH_POLICY_FULL_AND_PIECEWISE:
        os.environ.setdefault("INFINI_CUDAGRAPH_POLICY", CUDAGRAPH_POLICY_FULL_AND_PIECEWISE)
        if not os.environ.get("INFINI_FA_INGRAPH", "").strip():
            os.environ["INFINI_FA_INGRAPH"] = "decode"
            logger.info(
                "cudagraph_policy=full_and_piecewise → INFINI_FA_INGRAPH=decode"
            )
        return CUDAGRAPH_POLICY_FULL_AND_PIECEWISE
    return cudagraph_policy()
