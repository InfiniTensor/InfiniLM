# Copyright (c) 2025, InfiniCore
"""Environment flags for hybrid compiled prefill (Phases 3–6)."""

from __future__ import annotations

import os
from typing import List, Tuple

# Power-mode overflow buckets (+256 past anchor); ignored in linear mode.
COMPILE_OVERFLOW_BUCKET_1024 = 1280
COMPILE_OVERFLOW_BUCKET_4096 = 4352
COMPILE_OVERFLOW_BUCKET = COMPILE_OVERFLOW_BUCKET_4096

# Linear ladder step (tokens); default matches 2× paged block_size (256).
_DEFAULT_COMPILE_BUCKET_STEP = 512


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def prefill_compile_enabled() -> bool:
    """Master switch: torch compiled prefill + C++ decode (default off)."""
    return _truthy("INFINI_PREFILL_COMPILE", "0")


def prefill_cudagraph_enabled() -> bool:
    """Enable vLLM piecewise CUDAGraph on the compiled prefill backbone."""
    return _truthy("INFINI_PREFILL_CUDAGRAPH", "0")


def prefill_share_weights_enabled() -> bool:
    """Reuse C++ weight buffers for torch KV write in ``run_prefill_paged`` (opt-in)."""
    return _truthy("INFINI_PREFILL_SHARE_WEIGHTS", "0")


def return_logits_enabled() -> bool:
    """Opt-in: return pre-sample logits on CPU (``compile_prefill_parity.py`` only; default off)."""
    return _truthy("INFINI_RETURN_LOGITS", "0")


def compile_max_seq_len(default: int = 8448) -> int:
    raw = os.environ.get("INFINI_COMPILE_MAX_SEQ")
    return int(raw) if raw else default


def compile_bucket_mode() -> str:
    """``bench`` | ``linear`` | ``power`` (see ``compile_buckets``)."""
    return os.environ.get("COMPILE_BUCKET_MODE", "power").strip().lower()


def compile_bucket_step(default: int = _DEFAULT_COMPILE_BUCKET_STEP) -> int:
    raw = os.environ.get("COMPILE_BUCKET_STEP")
    return int(raw) if raw else default


def _compile_buckets_power(max_seq_len: int) -> Tuple[int, ...]:
    buckets: List[int] = [512, 1024]
    if max_seq_len > 1024:
        buckets.append(COMPILE_OVERFLOW_BUCKET_1024)
    if max_seq_len >= 4096:
        buckets.append(4096)
    if max_seq_len > 4096:
        buckets.append(COMPILE_OVERFLOW_BUCKET_4096)
    if max_seq_len >= 8192:
        buckets.append(8192)
    if max_seq_len >= 8448:
        buckets.append(8448)
    elif max_seq_len not in buckets:
        buckets.append(max_seq_len)
    return tuple(buckets)


# Sparse ladder for PRD-02 server TTFT sweeps (1024/4096/8192 + harness overflow).
_BENCH_BUCKET_ANCHORS: Tuple[int, ...] = (
    512,
    1024,
    1536,
    2048,
    4096,
    4608,
    5120,
    6144,
    7168,
    8192,
)


def _compile_buckets_bench(max_seq_len: int) -> Tuple[int, ...]:
    buckets = [a for a in _BENCH_BUCKET_ANCHORS if a <= max_seq_len]
    if max_seq_len not in buckets:
        buckets.append(max_seq_len)
    return tuple(buckets)


def _compile_buckets_linear(max_seq_len: int, step: int) -> Tuple[int, ...]:
    if step <= 0:
        raise ValueError(f"COMPILE_BUCKET_STEP must be positive, got {step}")
    buckets: List[int] = []
    b = step
    while b < max_seq_len:
        buckets.append(b)
        b += step
    if max_seq_len not in buckets:
        buckets.append(max_seq_len)
    return tuple(buckets)


def compile_buckets(max_seq_len: int) -> Tuple[int, ...]:
    """Runtime Inductor padding buckets (must match init warmup when unset)."""
    mode = compile_bucket_mode()
    if mode == "power":
        return _compile_buckets_power(max_seq_len)
    if mode in ("bench", "benchmark", "sparse"):
        return _compile_buckets_bench(max_seq_len)
    if mode in ("linear", "step"):
        return _compile_buckets_linear(max_seq_len, compile_bucket_step())
    raise ValueError(
        f"unknown COMPILE_BUCKET_MODE={mode!r} "
        "(use 'bench', 'linear', or 'power')"
    )


def compile_warmup_seq_lens(max_seq: int) -> List[int]:
    """Explicit Inductor warmup lengths (comma-separated ``COMPILE_WARMUP_SEQ_LENS``)."""
    raw = os.environ.get("COMPILE_WARMUP_SEQ_LENS")
    if raw:
        lens = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        lens = [8, 64, *compile_buckets(max_seq)]
    if max_seq not in lens:
        lens.append(max_seq)
    return sorted({n for n in lens if 0 < n <= max_seq})
