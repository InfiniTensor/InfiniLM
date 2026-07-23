# Copyright (c) 2025, InfiniCore
"""Python mirror of C++ ``cudagraph_dispatcher.hpp`` for unit tests / docs.

Runtime CG selection lives in InfiniLM RankWorker (C++). This module keeps the
descriptor→mode table testable without loading ``_infinilm``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Tuple


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


def _build_bs_to_padded_bucket(capture_sizes: List[int]) -> List[int]:
    """Mirror of ``piecewise_bucket_policy.hpp`` / ``env.build_bs_to_padded_bucket``."""
    if not capture_sizes:
        return [0]
    sizes = sorted({int(s) for s in capture_sizes if int(s) > 0}, reverse=True)
    max_capture_size = sizes[0]
    table = [0] * (max_capture_size + 1)
    for end, start in zip(sizes, sizes[1:] + [0]):
        for bs in range(start, end):
            table[bs] = start if bs == start else end
    table[max_capture_size] = max_capture_size
    return table


def _padded_bucket_for_seq_len(seq_len: int, bs_to_padded: List[int], fallback: int = 0) -> int:
    if 0 <= seq_len < len(bs_to_padded):
        padded = bs_to_padded[seq_len]
        if padded > 0:
            return padded
    return fallback


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
        self._bs_to_padded: List[int] = []
        self._max_capture: int = 0

    def _rebuild_pad_table(self) -> None:
        self._bs_to_padded = []
        self._max_capture = 0
        if not self.piecewise_keys:
            return
        caps = sorted(self.piecewise_keys)
        self._bs_to_padded = _build_bs_to_padded_bucket(caps)
        self._max_capture = caps[-1]

    def initialize_from_env(self) -> None:
        self.full_keys.clear()
        self.piecewise_keys.clear()
        self._bs_to_padded = []
        self._max_capture = 0
        policy = cudagraph_policy()
        if policy == "eager":
            return
        if policy in ("full_and_piecewise", ""):
            self.full_keys = _parse_csv_sizes(os.environ.get("INFINI_DECODE_CG_BATCHES"))
            self.piecewise_keys = _parse_csv_sizes(
                os.environ.get("INFINI_NATIVE_CG_CAPTURE_BUCKETS")
            )
            self._rebuild_pad_table()

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
        # Homogeneous single-req prefill with vLLM-style pad-up; MIXED / multi-req → NONE.
        if (
            not desc.uniform_decode
            and desc.num_reqs == 1
            and self.piecewise_keys
        ):
            if desc.num_tokens > self._max_capture:
                return CudaGraphRuntimeMode.NONE, desc
            padded = _padded_bucket_for_seq_len(
                desc.num_tokens, self._bs_to_padded, fallback=0
            )
            if padded > 0 and padded in self.piecewise_keys:
                key = BatchDescriptor(
                    num_tokens=padded,
                    num_reqs=desc.num_reqs,
                    uniform_decode=False,
                )
                return CudaGraphRuntimeMode.PIECEWISE, key
        return CudaGraphRuntimeMode.NONE, desc

    def none_reason(self, desc: BatchDescriptor, is_mixed: bool) -> str:
        if cudagraph_policy() == "eager":
            return "eager_policy"
        if is_mixed:
            return "mixed"
        if not desc.uniform_decode and desc.num_reqs > 1:
            return "multi_req_prefill"
        if desc.uniform_decode:
            return "decode_bs_miss"
        if self.piecewise_keys and desc.num_tokens > self._max_capture:
            return "over_max"
        return "bucket_miss"
