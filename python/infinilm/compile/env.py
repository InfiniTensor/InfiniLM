# Copyright (c) 2025, InfiniCore
"""Environment flags for hybrid compiled prefill (Phases 3–6).

Classification (for PR review):
  PROD — master switches + ladder driver:
    ``prefill_compile_enabled``, ``prefill_share_weights_enabled``, ``prefill_cudagraph_enabled``,
    ``compile_max_seq_len``, ``compile_buckets`` / ``compile_warmup_seq_lens``.
  DEBUG — diagnostics / smoke baselines only:
    ``prefill_cg_debug_ptrs_enabled``, ``prefill_cg_baseline_none``,
    ``return_logits_enabled``, ``INFINI_PREFILL_MEM_PROFILE`` (see ``mem_profile.py``).

Bucket ladders and CUDAGraph capture sizes are derived from ``INFINI_COMPILE_MAX_SEQ``
via ``vllm_unified_power_ladder`` (compile) and ``default_cudagraph_capture_buckets``
(CG capture excludes the 8448 overflow tail). KV-outside-graph, valid-seq scoping,
and vLLM garbage-tail staging are hardcoded in the compile path when CG is enabled.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

# Power-mode overflow buckets (+256 past anchor).
COMPILE_OVERFLOW_BUCKET_1024 = 1280
COMPILE_OVERFLOW_BUCKET_4096 = 4352
COMPILE_OVERFLOW_BUCKET_8192 = 8448
COMPILE_OVERFLOW_BUCKET = COMPILE_OVERFLOW_BUCKET_4096

# vLLM default piecewise ladder floor (512) through 8192, plus non-power max_seq tail.
_VLLM_POWER_LADDER_FLOOR = 512
_VLLM_POWER_LADDER_CAP = 8192


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def prefill_compile_enabled() -> bool:
    """Master switch: torch compiled prefill + C++ decode (default off)."""
    return _truthy("INFINI_PREFILL_COMPILE", "0")


def prefill_native_cg_enabled() -> bool:
    """Native C++ piecewise CUDAGraph prefill (HPCC v1; no torch.compile)."""
    return _truthy("INFINI_PREFILL_NATIVE_CG", "0")


def prefill_cudagraph_enabled() -> bool:
    """Enable vLLM piecewise CUDAGraph on the compiled prefill backbone."""
    return _truthy("INFINI_PREFILL_CUDAGRAPH", "0")


def prefill_cg_baseline_none() -> bool:
    """Smoke/parity: force ``CUDAGraphMode.NONE`` at replay (PIECEWISE vs NONE baseline)."""
    return _truthy("INFINI_PREFILL_CG_BASELINE_NONE", "0")


def prefill_share_weights_enabled() -> bool:
    """Reuse C++ weight buffers for torch KV write in ``run_prefill_paged`` (opt-in)."""
    return _truthy("INFINI_PREFILL_SHARE_WEIGHTS", "0")


def prefill_cg_debug_ptrs_enabled() -> bool:
    """Log pooled ``input_ids`` / ``slot_mapping`` ``data_ptr()`` at capture vs replay."""
    return _truthy("INFINI_PREFILL_CG_DEBUG_PTRS", "0")


def return_logits_enabled() -> bool:
    """Opt-in: return pre-sample logits on CPU (``compile_prefill_parity.py`` only; default off)."""
    return _truthy("INFINI_RETURN_LOGITS", "0")


def compile_max_seq_len(default: int = 8448) -> int:
    raw = os.environ.get("INFINI_COMPILE_MAX_SEQ")
    return int(raw) if raw else default


def compile_overflow_tail_bucket(max_seq_len: int) -> Optional[int]:
    """Headroom bucket when chat-template tokenization exceeds the 8192 compile cap."""
    if (
        max_seq_len >= _VLLM_POWER_LADDER_CAP
        and max_seq_len < COMPILE_OVERFLOW_BUCKET_8192
    ):
        return COMPILE_OVERFLOW_BUCKET_8192
    return None


def compile_bucket_ceiling(max_seq_len: int) -> int:
    """Upper bound for Inductor warmup + CG buffer sizing (includes overflow tail)."""
    tail = compile_overflow_tail_bucket(max_seq_len)
    return max(max_seq_len, tail or 0)


def vllm_unified_power_ladder(
    max_seq_len: int,
    *,
    min_bucket: int = _VLLM_POWER_LADDER_FLOOR,
) -> Tuple[int, ...]:
    """Unified Inductor pad buckets (vLLM power-of-2 + max_seq tail).

    Powers of two from ``min_bucket`` (512) through 8192. When ``max_seq_len`` exceeds
    8192 (chunked prefill not ready), append ``max_seq_len`` as the tail bucket
    (e.g. 8448 for bench harness overflow).
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    buckets: List[int] = []
    s = min_bucket
    while s <= min(max_seq_len, _VLLM_POWER_LADDER_CAP):
        buckets.append(s)
        next_s = s * 2
        if next_s <= s:
            break
        s = next_s
    if max_seq_len >= _VLLM_POWER_LADDER_CAP and _VLLM_POWER_LADDER_CAP not in buckets:
        buckets.append(_VLLM_POWER_LADDER_CAP)
    if max_seq_len > _VLLM_POWER_LADDER_CAP:
        buckets.append(max_seq_len)
    elif max_seq_len not in buckets:
        buckets.append(max_seq_len)
    tail = compile_overflow_tail_bucket(max_seq_len)
    if tail is not None:
        buckets.append(tail)
    ceiling = compile_bucket_ceiling(max_seq_len)
    return tuple(sorted(b for b in set(buckets) if b <= ceiling))


def default_cudagraph_capture_buckets(max_seq_len: int) -> Tuple[int, ...]:
    """Power ladder buckets eligible for PIECEWISE CUDAGraph capture.

    Caps at 8192: buckets above the power anchor (e.g. 8448 ``max_seq_len`` tail)
    use eager Inductor replay only.
    """
    ladder = vllm_unified_power_ladder(max_seq_len)
    return tuple(b for b in ladder if b <= _VLLM_POWER_LADDER_CAP)


def min_cudagraph_piecewise_bucket(capture_sizes: Optional[object]) -> int:
    """Smallest capture bucket eligible for PIECEWISE replay."""
    floor = _VLLM_POWER_LADDER_FLOOR
    if capture_sizes:
        return max(floor, min(int(b) for b in capture_sizes))
    return floor


def compile_buckets(max_seq_len: int) -> Tuple[int, ...]:
    """Runtime Inductor padding buckets (must match init warmup when unset)."""
    return vllm_unified_power_ladder(max_seq_len)


def build_bs_to_padded_bucket(capture_sizes: list[int]) -> list[int]:
    """O(1) seq_len → padded bucket table (mirrors vLLM ``init_with_cudagraph_sizes``).

    ``capture_sizes`` are the Inductor pad ladder (typically ``compile_buckets``).
    Returns ``table[bs]`` = smallest capture bucket ≥ ``bs`` within each interval.
    """
    if not capture_sizes:
        return [0]
    sizes = sorted({int(s) for s in capture_sizes if int(s) > 0}, reverse=True)
    max_capture_size = sizes[0]
    table = [0] * (max_capture_size + 1)
    for end, start in zip(sizes, sizes[1:] + [0]):
        for bs in range(start, end):
            if bs == start:
                table[bs] = start
            else:
                table[bs] = end
    table[max_capture_size] = max_capture_size
    return table


def padded_bucket_for_seq_len(
    seq_len: int,
    bs_to_padded: list[int],
    *,
    fallback: int,
) -> int:
    """Lookup padded bucket for ``seq_len`` using a vLLM-style pad table."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if seq_len < len(bs_to_padded):
        padded = bs_to_padded[seq_len]
        if padded > 0:
            return padded
    return fallback


def compile_warmup_seq_lens(max_seq: int) -> List[int]:
    """Explicit Inductor warmup lengths (comma-separated ``COMPILE_WARMUP_SEQ_LENS``)."""
    raw = os.environ.get("COMPILE_WARMUP_SEQ_LENS")
    if raw:
        lens = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        lens = [8, 64, *compile_buckets(max_seq)]
    ceiling = compile_bucket_ceiling(max_seq)
    if max_seq not in lens:
        lens.append(max_seq)
    if ceiling not in lens:
        lens.append(ceiling)
    return sorted({n for n in lens if 0 < n <= ceiling})
