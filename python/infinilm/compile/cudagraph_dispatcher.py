# Copyright (c) 2025, InfiniCore
"""Python mirror of C++ ``cudagraph_dispatcher.hpp`` for unit tests / docs.

Runtime CG selection lives in InfiniLM RankWorker (C++). This module keeps the
descriptor→mode table testable without loading ``_infinilm``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set, Tuple


class CudaGraphRuntimeMode(Enum):
    NONE = "NONE"
    PIECEWISE = "PIECEWISE"
    FULL = "FULL"


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int
    uniform_decode: bool


def _parse_csv_sizes(raw: Optional[str]) -> Set[int]:
    if not raw or not str(raw).strip():
        return set()
    out: Set[int] = set()
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return out


def cudagraph_policy() -> str:
    raw = os.environ.get("INFINI_CUDAGRAPH_POLICY", "eager").strip().lower()
    if raw in ("", "eager"):
        return "eager"
    if raw == "full_and_piecewise":
        return "full_and_piecewise"
    return ""


class CudagraphDispatcher:
    """FULL / PIECEWISE / NONE selector matching C++ ``CudagraphDispatcher``."""

    def __init__(self) -> None:
        self.full_keys: Set[int] = set()
        self.piecewise_keys: Set[int] = set()

    def initialize_from_env(self) -> None:
        self.full_keys.clear()
        self.piecewise_keys.clear()
        policy = cudagraph_policy()
        if policy == "eager":
            return
        if policy in ("full_and_piecewise", ""):
            self.full_keys = _parse_csv_sizes(os.environ.get("INFINI_DECODE_CG_BATCHES"))
            self.piecewise_keys = _parse_csv_sizes(
                os.environ.get("INFINI_NATIVE_CG_CAPTURE_BUCKETS")
            )

    def dispatch(
        self, desc: BatchDescriptor
    ) -> Tuple[CudaGraphRuntimeMode, BatchDescriptor]:
        if cudagraph_policy() == "eager":
            return CudaGraphRuntimeMode.NONE, desc
        if desc.uniform_decode and desc.num_tokens in self.full_keys:
            key = BatchDescriptor(
                num_tokens=desc.num_tokens,
                num_reqs=desc.num_tokens,
                uniform_decode=True,
            )
            return CudaGraphRuntimeMode.FULL, key
        # Homogeneous single-req prefill bucket hit; MIXED / multi-req → NONE.
        if (
            not desc.uniform_decode
            and desc.num_reqs == 1
            and desc.num_tokens in self.piecewise_keys
        ):
            return CudaGraphRuntimeMode.PIECEWISE, desc
        return CudaGraphRuntimeMode.NONE, desc
