# Copyright (c) 2025, InfiniCore
"""Environment flags for hybrid compiled prefill (Phases 3–6).

Classification (for PR review):
  PROD — master switches + ladder driver (see docs/INFINI_FLAGS.md):
    ``prefill_native_cg_enabled``, ``prefill_chunked_enabled``, ``prefill_chunk_size``,
    ``prefill_compile_enabled``, ``prefill_share_weights_enabled``, ``prefill_cudagraph_enabled``,
    ``compile_max_seq_len``, ``compile_buckets`` / ``compile_warmup_seq_lens``,
    ``v1_scheduler_enabled``, ``schedule_homogeneous_enabled`` (deprecated no-op).
  C++ code defaults (no env): decode pre-barrier on, piecewise post-AR barrier trim
    (opt-out: INFINI_DECODE_SKIP_PRE_BARRIER, INFINI_PIECEWISE_KEEP_BARRIERS).
  C++ production env (``infinilm_production_env.sh``): ``INFINI_DECODE_LIGHT_SYNC``,
    ``INFINI_NATIVE_CG_CAPTURE_BUCKETS`` — also read here for Inductor bootstrap /
    pad ladders so Python matches C++ ``PiecewisePrefillCompiler``.
  DEBUG — diagnostics / smoke baselines only:
    ``prefill_cg_debug_ptrs_enabled``, ``prefill_cg_baseline_none``,
    ``return_logits_enabled``, ``INFINI_PREFILL_MEM_PROFILE`` (see ``mem_profile.py``).

When ``INFINI_NATIVE_CG_CAPTURE_BUCKETS`` is set, Inductor bootstrap and
``compile_buckets`` use that list only (no auto power-of-two through 8192).
Otherwise ladders fall back to ``vllm_unified_power_ladder`` /
``default_cudagraph_capture_buckets``. KV-outside-graph, valid-seq scoping,
and vLLM garbage-tail staging are hardcoded in the compile path when CG is enabled.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_PREFILL_COMPILE_WARNED = False
_SCHEDULE_HOMOGENEOUS_WARNED = False

# Power-mode overflow buckets (+256 past anchor).
COMPILE_OVERFLOW_BUCKET_1024 = 1280
COMPILE_OVERFLOW_BUCKET_4096 = 4352
COMPILE_OVERFLOW_BUCKET_8192 = 8448
COMPILE_OVERFLOW_BUCKET = COMPILE_OVERFLOW_BUCKET_4096

# vLLM default piecewise ladder floor (512) through 8192, plus non-power max_seq tail.
_VLLM_POWER_LADDER_FLOOR = 512
_VLLM_POWER_LADDER_CAP = 8192

# Track-B MoE AOT step totals (pack_moe DEFAULT_BUCKETS + decode sub-16 powers).
_MOE_AOT_STEP_LADDER_MAX = 4096
_MOE_AOT_STEP_TOKEN_LADDER: Tuple[int, ...] = tuple(
    1 << i for i in range(0, 13) if (1 << i) <= _MOE_AOT_STEP_LADDER_MAX
)


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no", "off")


def prefill_compile_enabled() -> bool:
    """Deprecated PRD-02 torch.compile prefill path (removed). Use INFINI_PREFILL_NATIVE_CG=1."""
    global _PREFILL_COMPILE_WARNED
    if _truthy("INFINI_PREFILL_COMPILE", "0"):
        if not _PREFILL_COMPILE_WARNED:
            logger.warning(
                "INFINI_PREFILL_COMPILE is deprecated and ignored; "
                "use INFINI_PREFILL_NATIVE_CG=1 for native C++ piecewise CG"
            )
            _PREFILL_COMPILE_WARNED = True
    return False


def prefill_native_cg_enabled() -> bool:
    """Native C++ piecewise CUDAGraph prefill (HPCC v1; no torch.compile)."""
    return _truthy("INFINI_PREFILL_NATIVE_CG", "0")


def prefill_chunked_enabled() -> bool:
    """Multi-step chunked prefill for long prompts (server / AsyncLLMEngine)."""
    return _truthy("INFINI_PREFILL_CHUNKED", "0")


def prefill_chunk_size(default: int = 8192) -> int:
    """Max new tokens per chunked prefill step (clamped to power ladder cap)."""
    raw = os.environ.get("INFINI_PREFILL_CHUNK_SIZE")
    size = int(raw) if raw else default
    return min(max(size, 1), _VLLM_POWER_LADDER_CAP)


def v1_scheduler_enabled() -> bool:
    """Master switch: vLLM 0.17 token-budget scheduler (default off)."""
    return _truthy("INFINI_V1_SCHEDULER", "0")


def schedule_homogeneous_enabled() -> bool:
    """Deprecated: homogeneous PREFILL XOR DECODE is removed (vLLM-align M6).

    If ``INFINI_SCHEDULE_HOMOGENEOUS`` is set truthy, log a one-shot warning and
    ignore — v1 always uses continuous mixed scheduling + MoE shape gate.
    """
    global _SCHEDULE_HOMOGENEOUS_WARNED
    if _truthy("INFINI_SCHEDULE_HOMOGENEOUS", "0"):
        if not _SCHEDULE_HOMOGENEOUS_WARNED:
            logger.warning(
                "INFINI_SCHEDULE_HOMOGENEOUS is deprecated and ignored; "
                "v1 scheduler always uses continuous mixed batching "
                "(see docs/M6_scheduler_vllm_align_design.md)"
            )
            _SCHEDULE_HOMOGENEOUS_WARNED = True
    return False


def max_num_batched_tokens(default: int = 8192) -> int:
    """Token budget per v1 scheduler step (primary shared chunker, as vLLM)."""
    raw = os.environ.get("INFINI_MAX_NUM_BATCHED_TOKENS")
    return int(raw) if raw else default


def long_prefill_threshold(default: int = 0) -> int:
    """Per-request prefill cap for one v1 step (vLLM ``long_prefill_token_threshold``).

    Default **0** (inactive) when unset. Explicit ``INFINI_LONG_PREFILL_THRESHOLD``
    wins; otherwise chunked prefill uses ``INFINI_PREFILL_CHUNK_SIZE`` as the
    per-request cap only (shared step budget remains ``max_num_batched_tokens``).
    """
    raw = os.environ.get("INFINI_LONG_PREFILL_THRESHOLD")
    if raw:
        return int(raw)
    if prefill_chunked_enabled():
        return prefill_chunk_size()
    return default


def moe_aot_step_max_tokens() -> int:
    """Largest MoE/piecewise AOT bucket (shape-gate ceiling)."""
    return _MOE_AOT_STEP_LADDER_MAX


def moe_aot_step_token_ladder() -> Tuple[int, ...]:
    """MoE/piecewise AOT bucket sizes (powers of two through 4096)."""
    return _MOE_AOT_STEP_TOKEN_LADDER


def moe_aot_step_total_allowed(total: int) -> bool:
    """True if step total can be served by MoE AOT (runtime pads up to a bucket).

    Rejects totals above the max compiled bucket (e.g. 4097). Scheduler does not
    invent pad tokens; engine pads within 1..max. Exact power-of-two is preferred
    but not required so final prefill remainders still schedule.
    """
    return 0 < total <= _MOE_AOT_STEP_LADDER_MAX


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


def torch_compile_enabled() -> bool:
    """Master switch: unified prefill + decode via torch.compile (PRD-04)."""
    return _truthy("INFINI_TORCH_COMPILE", "0")


def torch_compile_share_weights_enabled() -> bool:
    """Reuse C++ weight buffers for torch compile path (both phases)."""
    return _truthy("INFINI_TORCH_COMPILE_SHARE_WEIGHTS", "0")


def torch_compile_cache_root() -> str:
    """Inductor + metadata cache root for PRD-04 torch.compile."""
    return os.environ.get(
        "INFINI_TORCH_COMPILE_CACHE",
        "bench_results/torch_compile_cache",
    )


def piecewise_inductor_cache_root() -> str:
    """AOTInductor artifact cache for M4 piecewise kernel segments."""
    return os.environ.get(
        "INFINI_PIECEWISE_INDUCTOR_CACHE",
        "bench_results/piecewise_inductor_cache",
    )


def piecewise_inductor_require_aot() -> bool:
    """When set, AOT packaging failures must not fall back to torch.compile."""
    return _truthy("INFINI_PIECEWISE_INDUCTOR_REQUIRE_AOT", "0")


_COMPILE_ON_MISS_WARNED = False


def piecewise_inductor_compile_on_miss() -> bool:
    """Deprecated: InferEngine never compile-on-miss.

    Default is off. If ``INFINI_PIECEWISE_INDUCTOR_COMPILE_ON_MISS=1``, emit a
    one-shot DeprecationWarning and ignore (always returns False). Use
    ``python -m infinilm.server.entry --phase compile|all`` for offline AOT.
    """
    global _COMPILE_ON_MISS_WARNED
    raw = os.environ.get("INFINI_PIECEWISE_INDUCTOR_COMPILE_ON_MISS", "0")
    if str(raw).strip().lower() not in ("", "0", "false", "no", "off"):
        if not _COMPILE_ON_MISS_WARNED:
            import warnings

            warnings.warn(
                "INFINI_PIECEWISE_INDUCTOR_COMPILE_ON_MISS=1 is deprecated and ignored; "
                "InferEngine is register-only. Use "
                "`python -m infinilm.server.entry --phase compile|all` for offline AOT.",
                DeprecationWarning,
                stacklevel=2,
            )
            _COMPILE_ON_MISS_WARNED = True
    return False


def aot_check_skip() -> bool:
    """When set, skip piecewise AOT register/check at InferEngine bootstrap (debug)."""
    return _truthy("INFINI_AOT_CHECK_SKIP", "0")


def piecewise_inductor_segment_enabled() -> bool:
    """Use AOTInductor kernels inside native piecewise pre/post segments (M4)."""
    return _truthy("INFINI_PIECEWISE_INDUCTOR_SEGMENT", "0")


_TORCH_COMPILE_MUTEX_WARNED = False


def check_torch_compile_mutual_exclusion() -> None:
    """Log error when PRD-03 native CG and PRD-04 torch.compile are both enabled."""
    global _TORCH_COMPILE_MUTEX_WARNED
    if torch_compile_enabled() and prefill_native_cg_enabled():
        if not _TORCH_COMPILE_MUTEX_WARNED:
            logger.error(
                "INFINI_TORCH_COMPILE=1 and INFINI_PREFILL_NATIVE_CG=1 are mutually "
                "exclusive; undefined dispatch if both are set at server init"
            )
            _TORCH_COMPILE_MUTEX_WARNED = True


def compile_max_seq_len(default: int = 8192) -> int:
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
    8192, append ``max_seq_len`` as the tail bucket (e.g. 8448 for bench harness
    overflow). Long prompts beyond 8192 use chunked prefill (``INFINI_PREFILL_CHUNKED``).
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


def vllm_capture_ladder_enabled() -> bool:
    """Merge sub-512 power ladder into native CG capture (vLLM 0.20 piecewise policy)."""
    return _truthy("INFINI_VLLM_CAPTURE_LADDER", "0")


def vllm_piecewise_capture_sizes(chunk_cap: int = 512) -> Tuple[int, ...]:
    """Powers of two from 1 through ``min(chunk_cap, 512)`` (vLLM tail-chunk ladder)."""
    if chunk_cap <= 0:
        raise ValueError(f"chunk_cap must be positive, got {chunk_cap}")
    cap = min(int(chunk_cap), _VLLM_POWER_LADDER_FLOOR)
    buckets: List[int] = []
    s = 1
    while s <= cap:
        buckets.append(s)
        next_s = s * 2
        if next_s <= s:
            break
        s = next_s
    return tuple(buckets)


def native_piecewise_capture_buckets_vllm(
    max_seq_len: int,
    chunk_cap: int = 512,
) -> Tuple[int, ...]:
    """Sub-512 ladder merged with power ladder 512..8192 for CG capture."""
    sub = vllm_piecewise_capture_sizes(chunk_cap)
    power = vllm_unified_power_ladder(max_seq_len)
    merged = tuple(sorted(set(sub) | set(power)))
    return tuple(b for b in merged if b <= _VLLM_POWER_LADDER_CAP)


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


def native_cg_capture_buckets_from_env() -> Optional[Tuple[int, ...]]:
    """Parse ``INFINI_NATIVE_CG_CAPTURE_BUCKETS`` (same CSV as C++ capture list).

    When set, InfiniLM must not invent extra vLLM power-ladder buckets (e.g. B8192)
    for Inductor bootstrap / compile-on-miss — that list is the authoritative set.
    """
    raw = os.environ.get("INFINI_NATIVE_CG_CAPTURE_BUCKETS")
    if raw is None or not str(raw).strip():
        return None
    buckets: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            buckets.append(int(token))
    if not buckets:
        return None
    return tuple(sorted(set(buckets)))


def compile_buckets(max_seq_len: int) -> Tuple[int, ...]:
    """Runtime Inductor padding buckets (must match init warmup when unset)."""
    override = native_cg_capture_buckets_from_env()
    if override is not None:
        return override
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


def graph_replay_bucket_for_seq_len(
    seq_len: int,
    bs_to_padded: list[int],
    *,
    fallback: int,
) -> int:
    """CG replay bucket matches compile pad (includes 8448 overflow tail)."""
    return padded_bucket_for_seq_len(seq_len, bs_to_padded, fallback=fallback)


def native_piecewise_capture_buckets(max_seq_len: int) -> Tuple[int, ...]:
    """Native capture / Inductor bootstrap buckets.

    Prefer ``INFINI_NATIVE_CG_CAPTURE_BUCKETS`` when set (parity with C++).
    Otherwise: optional vLLM sub-512 merge, else power buckets through 8192.
    """
    override = native_cg_capture_buckets_from_env()
    if override is not None:
        return override
    if vllm_capture_ladder_enabled():
        return native_piecewise_capture_buckets_vllm(max_seq_len, prefill_chunk_size(default=512))
    return default_cudagraph_capture_buckets(max_seq_len)


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
