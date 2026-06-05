# Copyright (c) 2025, InfiniCore
"""Environment flags for hybrid compiled prefill (Phases 3–6).

Classification (for PR review):
  PROD — default-on production path when ``INFINI_PREFILL_COMPILE=1`` (+ share/CG flags):
    ``prefill_compile_enabled``, ``prefill_share_weights_enabled``, ``prefill_cudagraph_enabled``,
    ``compile_max_seq_len``, ``compile_bucket_mode`` (``power`` in serve scripts),
    ``compile_buckets`` / ``compile_warmup_seq_lens``.
  REPRO — opt-in experiment / bisect knobs (not set in ``serve_infinilm.sh``):
    ``prefill_cg_kv_outside_graph``, ``prefill_cg_allow_partial_pad``,
    ``prefill_cg_piecewise_min_bucket``,
    ``prefill_cudagraph_max_bucket``,
    ``prefill_cudagraph_capture_buckets`` / ``CAPTURE_MODE=longseq``,
    ``INFINI_PREFILL_CG_POWER_LADDER``, ``INFINI_PREFILL_CG_POOL_TIER_ISOLATION``,
    ``COMPILE_BUCKET_MODE=power|linear``, ``COMPILE_WARMUP_SEQ_LENS``.
  DEBUG — diagnostics only:
    ``prefill_cg_debug_ptrs_enabled``, ``return_logits_enabled``,
    ``INFINI_PREFILL_MEM_PROFILE`` (see ``mem_profile.py``).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

# Power-mode overflow buckets (+256 past anchor); ignored in linear mode.
COMPILE_OVERFLOW_BUCKET_1024 = 1280
COMPILE_OVERFLOW_BUCKET_4096 = 4352
COMPILE_OVERFLOW_BUCKET_8192 = 8448
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


def prefill_cg_kv_outside_graph() -> bool:
    """Phase 1: stage K/V in-graph; flush to paged cache eagerly after CUDAGraph replay.

    Default off until Phase 1 validation passes. Set ``INFINI_PREFILL_CG_KV_OUTSIDE_GRAPH=1``.
    """
    return _truthy("INFINI_PREFILL_CG_KV_OUTSIDE_GRAPH", "0")


def prefill_cg_allow_partial_pad() -> bool:
    """When True, partial bucket pad uses PIECEWISE CUDAGraph replay.

    Default **off**: MetaX mcblas can ATU on vLLM-bench random prompts @519→1024
    during partial PIECEWISE replay. Full-bucket replay stays PIECEWISE.
    """
    return _truthy("INFINI_PREFILL_CG_ALLOW_PARTIAL", "0")


def prefill_cg_piecewise_min_bucket(default: int = 512) -> int:
    """Minimum capture bucket eligible for PIECEWISE CUDAGraph replay.

    Set ``INFINI_PREFILL_CG_PIECEWISE_MIN_BUCKET`` (default **512**). Buckets below
    this floor use eager Inductor even when present in ``cudagraph_capture_sizes``.
    """
    raw = os.environ.get("INFINI_PREFILL_CG_PIECEWISE_MIN_BUCKET")
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def prefill_cg_baseline_none() -> bool:
    """Smoke/parity: force ``CUDAGraphMode.NONE`` at replay (PIECEWISE vs NONE baseline)."""
    return _truthy("INFINI_PREFILL_CG_BASELINE_NONE", "0")


def prefill_cg_valid_seq_len() -> bool:
    """Phase 2: pass pooled ``valid_seq_len`` into ``prefill_flash_attention`` for partial replay.

    Default on when ``INFINI_PREFILL_CG_VALID_SEQ_LEN`` is unset (safe no-op at full bucket).
    Set ``INFINI_PREFILL_CG_VALID_SEQ_LEN=0`` to disable.
    """
    return _truthy("INFINI_PREFILL_CG_VALID_SEQ_LEN", "1")


def prefill_cg_skip_poison_recapture() -> bool:
    """Smoke/parity: skip partial poison marking and ladder recapture (graph reuse stress).

    Set ``INFINI_PREFILL_CG_SKIP_POISON_RECAPTURE=1`` only in validation smokes — not for serve.
    """
    return _truthy("INFINI_PREFILL_CG_SKIP_POISON_RECAPTURE", "0")


def prefill_cudagraph_max_bucket(default: int = 4096) -> Optional[int]:
    """Max bucket for CUDAGraph capture/replay; larger buckets use eager Inductor.

    Set ``INFINI_PREFILL_CUDAGRAPH_MAX_BUCKET=0`` to disable the cap (capture all
    buckets). Default 4096 avoids @8192+ replay ATU faults on MetaX.
    Ignored when explicit capture buckets / ``CAPTURE_MODE=longseq`` is set.
    """
    if os.environ.get("INFINI_PREFILL_CUDAGRAPH_CAPTURE_BUCKETS", "").strip():
        return None
    if os.environ.get("INFINI_PREFILL_CUDAGRAPH_CAPTURE_MODE", "").strip().lower() in (
        "longseq",
        "long_seq",
        "large",
        "8192",
    ):
        return None
    raw = os.environ.get("INFINI_PREFILL_CUDAGRAPH_MAX_BUCKET")
    if raw is None:
        return default if prefill_cudagraph_enabled() else None
    val = raw.strip().lower()
    if val in ("", "0", "none", "off", "false", "no"):
        return None
    return int(raw)


# Large-bucket-only capture (saves init vs full bench ladder; ATU experiment).
_LONGSEQ_CUDAGRAPH_CAPTURE_BUCKETS: Tuple[int, ...] = (7168, 8192, 8448)


def prefill_cudagraph_capture_buckets(max_seq_len: int) -> Optional[Tuple[int, ...]]:
    """Explicit CUDAGraph capture sizes (Inductor still warms all compile buckets).

    ``INFINI_PREFILL_CUDAGRAPH_CAPTURE_MODE=longseq`` → 7168,8192,8448 (within max_seq).
    Or ``INFINI_PREFILL_CUDAGRAPH_CAPTURE_BUCKETS=7168,8192,8448``.
    """
    mode = os.environ.get("INFINI_PREFILL_CUDAGRAPH_CAPTURE_MODE", "").strip().lower()
    raw = os.environ.get("INFINI_PREFILL_CUDAGRAPH_CAPTURE_BUCKETS", "").strip()
    if mode in ("longseq", "long_seq", "large", "8192"):
        candidates = _LONGSEQ_CUDAGRAPH_CAPTURE_BUCKETS
    elif raw:
        candidates = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    else:
        return None
    allowed = set(compile_buckets(max_seq_len))
    picked = tuple(sorted(b for b in candidates if b in allowed and b <= max_seq_len))
    return picked if picked else None


def prefill_cg_pool_tier_isolation() -> bool:
    """Phase 3b: separate vLLM CUDAGraph memory pools per power-tier (short/mid/long)."""
    return _truthy("INFINI_PREFILL_CG_POOL_TIER_ISOLATION", "0")


# Legacy sparse CG anchors (repro only; production uses ``vllm_unified_power_ladder``).
POWER_CG_CAPTURE_ANCHORS: Tuple[int, ...] = (1024, 4096, 8192)

# Pool tier boundaries aligned to power anchors (pad bucket → tier).
_CG_POOL_TIER_SHORT_MAX = 1024
_CG_POOL_TIER_MID_MAX = 4096

# vLLM default piecewise ladder floor (512) through 8192, plus non-power ``max_seq_len`` tail.
_VLLM_POWER_LADDER_FLOOR = 512
_VLLM_POWER_LADDER_CAP = 8192


def compile_overflow_tail_bucket(max_seq_len: int) -> Optional[int]:
    """Headroom bucket when chat-template tokenization exceeds the 8192 compile cap."""
    if (
        max_seq_len >= _VLLM_POWER_LADDER_CAP
        and max_seq_len < COMPILE_OVERFLOW_BUCKET_8192
    ):
        return COMPILE_OVERFLOW_BUCKET_8192
    return None


def compile_bucket_ceiling(max_seq_len: int) -> int:
    """Upper bound for Inductor warmup + CG capture buckets (includes overflow tail)."""
    tail = compile_overflow_tail_bucket(max_seq_len)
    return max(max_seq_len, tail or 0)


def vllm_unified_power_ladder(
    max_seq_len: int,
    *,
    min_bucket: int = _VLLM_POWER_LADDER_FLOOR,
) -> Tuple[int, ...]:
    """Unified Inductor pad + CUDAGraph capture buckets (vLLM power-of-2 + max_seq tail).

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


def prefill_cg_power_ladder_enabled() -> bool:
    """Legacy repro knob (unified power ladder is default via ``COMPILE_BUCKET_MODE=power``).

    When set with ``COMPILE_BUCKET_MODE=bench``, forces CG capture sizes to
    ``vllm_unified_power_ladder`` while runtime padding stays on the bench ladder.
    """
    return _truthy("INFINI_PREFILL_CG_POWER_LADDER", "0")


def prefill_cg_power_capture_buckets(max_seq_len: int) -> Tuple[int, ...]:
    """CUDAGraph capture sizes aligned with the unified vLLM power ladder."""
    return vllm_unified_power_ladder(max_seq_len)


def cudagraph_pool_tier_id(bucket: int) -> str:
    """Map a compile/capture bucket to short | mid | long (power-tier graph pool)."""
    if bucket <= _CG_POOL_TIER_SHORT_MAX:
        return "short"
    if bucket <= _CG_POOL_TIER_MID_MAX:
        return "mid"
    return "long"


def cudagraph_pool_tier_repr_bucket(tier_id: str) -> int:
    """Representative bucket per tier for pool (re)initialization."""
    return {"short": 512, "mid": 2048, "long": 8192}[tier_id]


def cudagraph_poison_ladder_buckets(
    bucket: int,
    capture_sizes: Optional[object],
) -> frozenset[int]:
    """Capture buckets that may be poisoned after partial replay at ``bucket``."""
    capture_set = set(capture_sizes or ())
    if not capture_set:
        return frozenset()
    if prefill_cg_pool_tier_isolation():
        # Partial replay poisons the shared tier graph pool (512+1024 share "short").
        tier = cudagraph_pool_tier_id(bucket)
        return frozenset(
            b for b in capture_set if cudagraph_pool_tier_id(b) == tier
        )
    return frozenset({bucket} | {b for b in capture_set if b > bucket})


def cudagraph_buckets_needing_recapture(
    needs_reprime: set[int],
    bucket: int,
    capture_sizes: Optional[object],
) -> Tuple[int, ...]:
    """Buckets to re-capture before forward at ``bucket`` (tier-scoped when isolation on)."""
    capture_set = set(capture_sizes or ())
    if not needs_reprime or not capture_set:
        return ()
    if prefill_cg_pool_tier_isolation():
        tier = cudagraph_pool_tier_id(bucket)
        tier_capture = sorted(
            b for b in capture_set if cudagraph_pool_tier_id(b) == tier
        )
        if not any(b in needs_reprime for b in tier_capture):
            return ()
        # Re-capture every bucket in the tier: pools are shared within a tier.
        return tuple(tier_capture)
    if bucket not in needs_reprime:
        return ()
    return tuple(sorted(b for b in needs_reprime if b in capture_set and b >= bucket))


def min_cudagraph_piecewise_bucket(capture_sizes: Optional[object]) -> int:
    """Smallest capture bucket eligible for PIECEWISE replay (env floor + capture set)."""
    floor = prefill_cg_piecewise_min_bucket()
    if capture_sizes:
        return max(floor, min(int(b) for b in capture_sizes))
    return floor


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


def compile_bucket_mode() -> str:
    """``bench`` | ``linear`` | ``power`` (see ``compile_buckets``)."""
    return os.environ.get("COMPILE_BUCKET_MODE", "power").strip().lower()


def compile_bucket_step(default: int = _DEFAULT_COMPILE_BUCKET_STEP) -> int:
    raw = os.environ.get("COMPILE_BUCKET_STEP")
    return int(raw) if raw else default


def _compile_buckets_power(max_seq_len: int) -> Tuple[int, ...]:
    return vllm_unified_power_ladder(max_seq_len)


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
    ceiling = compile_bucket_ceiling(max_seq)
    if max_seq not in lens:
        lens.append(max_seq)
    if ceiling not in lens:
        lens.append(ceiling)
    return sorted({n for n in lens if 0 < n <= ceiling})
