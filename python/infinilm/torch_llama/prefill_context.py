# Copyright (c) 2025, InfiniCore
"""Thread-local prefill metadata for compile-friendly attention (valid seq len)."""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import torch

_TLS = threading.local()


@dataclass
class PrefillCompileContext:
    """Valid token count when inputs are padded to a compile bucket."""

    valid_seq_len: int


def active_prefill_compile_context() -> Optional[PrefillCompileContext]:
    return getattr(_TLS, "ctx", None)


@contextlib.contextmanager
def prefill_compile_context(
    valid_seq_len: int,
) -> Iterator[PrefillCompileContext]:
    prev = getattr(_TLS, "ctx", None)
    ctx = PrefillCompileContext(valid_seq_len=int(valid_seq_len))
    _TLS.ctx = ctx
    try:
        yield ctx
    finally:
        _TLS.ctx = prev


def valid_seq_len_tensor(
    query: torch.Tensor,
    *,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Scalar int64 tensor ``[1]`` for ``prefill_flash_attention`` when padded."""
    ctx = active_prefill_compile_context()
    if ctx is None:
        return None
    bucket_len = int(query.shape[1])
    if ctx.valid_seq_len >= bucket_len:
        return None
    dev = device or query.device
    return torch.tensor([ctx.valid_seq_len], dtype=torch.int64, device=dev)
