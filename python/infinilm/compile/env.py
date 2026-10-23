"""CUDA-graph policy + continuous-batching / chunked-prefill env knobs."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CUDAGRAPH_POLICY_EAGER = "eager"
CUDAGRAPH_POLICY_FULL_AND_PIECEWISE = "full_and_piecewise"
_CUDAGRAPH_POLICIES = frozenset(
    {CUDAGRAPH_POLICY_EAGER, CUDAGRAPH_POLICY_FULL_AND_PIECEWISE}
)

_SCHEDULE_HOMOGENEOUS_WARNED = False
_PREFILL_CHUNK_SIZE_WARNED = False

# MoE/piecewise AOT step ladder (powers of two through 4096).
_MOE_AOT_STEP_LADDER_MAX = 4096
_MOE_AOT_STEP_TOKEN_LADDER: Tuple[int, ...] = tuple(
    1 << i for i in range(0, 13) if (1 << i) <= _MOE_AOT_STEP_LADDER_MAX
)


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


def _setdefault_env(name: str, value: str) -> None:
    """Set env only when unset or empty."""
    cur = os.environ.get(name)
    if cur is None or not str(cur).strip():
        os.environ[name] = value


def cudagraph_policy() -> str:
    raw = os.environ.get("INFINI_CUDAGRAPH_POLICY", "").strip().lower()
    if raw in ("", CUDAGRAPH_POLICY_EAGER):
        return CUDAGRAPH_POLICY_EAGER if raw == CUDAGRAPH_POLICY_EAGER else ""
    if raw == CUDAGRAPH_POLICY_FULL_AND_PIECEWISE:
        return CUDAGRAPH_POLICY_FULL_AND_PIECEWISE
    return ""


def apply_cudagraph_policy_env(policy: Optional[str] = None) -> str:
    """Expand companion knobs when F&P is selected.

    Keeps existing FA_INGRAPH=decode setdefault, and setdefaults
    ``INFINI_MAX_NUM_BATCHED_TOKENS=2048`` under F&P when unset.
    """
    pol = (policy or os.environ.get("INFINI_CUDAGRAPH_POLICY", "")).strip().lower()
    if pol == CUDAGRAPH_POLICY_FULL_AND_PIECEWISE:
        os.environ.setdefault(
            "INFINI_CUDAGRAPH_POLICY", CUDAGRAPH_POLICY_FULL_AND_PIECEWISE
        )
        if not os.environ.get("INFINI_FA_INGRAPH", "").strip():
            os.environ["INFINI_FA_INGRAPH"] = "decode"
            logger.info(
                "cudagraph_policy=full_and_piecewise → INFINI_FA_INGRAPH=decode"
            )
        _setdefault_env("INFINI_MAX_NUM_BATCHED_TOKENS", "2048")
        return CUDAGRAPH_POLICY_FULL_AND_PIECEWISE
    return cudagraph_policy()


def prefill_native_cg_enabled() -> bool:
    """Native C++ piecewise CUDAGraph prefill (follows F&P policy when set)."""
    raw = os.environ.get("INFINI_CUDAGRAPH_POLICY", "").strip().lower()
    if raw in _CUDAGRAPH_POLICIES:
        return raw == CUDAGRAPH_POLICY_FULL_AND_PIECEWISE
    return _truthy("INFINI_PREFILL_NATIVE_CG", "0")


def prefill_chunked_enabled() -> bool:
    """Multi-step chunked prefill for long prompts."""
    return _truthy("INFINI_PREFILL_CHUNKED", "0")


def max_num_batched_tokens(default: int = 8192) -> int:
    """Token budget per v1 scheduler step (shared chunker / step budget).

    Canonical: ``INFINI_MAX_NUM_BATCHED_TOKENS``. Legacy
    ``INFINILM_MAX_NUM_BATCHED_TOKENS`` and deprecated
    ``INFINI_PREFILL_CHUNK_SIZE`` are one-release shims when MAX is unset.
    """
    global _PREFILL_CHUNK_SIZE_WARNED
    raw_max = os.environ.get("INFINI_MAX_NUM_BATCHED_TOKENS")
    raw_legacy = os.environ.get("INFINILM_MAX_NUM_BATCHED_TOKENS")
    raw_chunk = os.environ.get("INFINI_PREFILL_CHUNK_SIZE")

    if raw_chunk and not _PREFILL_CHUNK_SIZE_WARNED:
        logger.warning(
            "INFINI_PREFILL_CHUNK_SIZE is deprecated; use "
            "INFINI_MAX_NUM_BATCHED_TOKENS (shared step budget). "
            "CHUNK is ignored when MAX_BATCHED is set; if only CHUNK is set it "
            "is treated as MAX_BATCHED for one release."
        )
        _PREFILL_CHUNK_SIZE_WARNED = True

    if raw_max:
        max_v = int(raw_max)
        if raw_chunk and int(raw_chunk) != max_v:
            logger.warning(
                "INFINI_PREFILL_CHUNK_SIZE=%s ignored; "
                "INFINI_MAX_NUM_BATCHED_TOKENS=%s wins",
                raw_chunk,
                raw_max,
            )
        return max(1, max_v)

    if raw_legacy:
        return max(1, int(raw_legacy))

    if raw_chunk:
        return max(1, int(raw_chunk))

    if prefill_chunked_enabled() and prefill_native_cg_enabled():
        return 2048
    return default


def prefill_chunk_size(default: int = 8192) -> int:
    """Deprecated alias of ``max_num_batched_tokens`` (one shared step budget)."""
    return max_num_batched_tokens(default=default)


def v1_scheduler_enabled() -> bool:
    """Master switch: token-budget continuous mixed scheduler (default off)."""
    return _truthy("INFINI_V1_SCHEDULER", "0")


def schedule_homogeneous_enabled() -> bool:
    """Deprecated: homogeneous PREFILL XOR DECODE is removed."""
    global _SCHEDULE_HOMOGENEOUS_WARNED
    if _truthy("INFINI_SCHEDULE_HOMOGENEOUS", "0"):
        if not _SCHEDULE_HOMOGENEOUS_WARNED:
            logger.warning(
                "INFINI_SCHEDULE_HOMOGENEOUS is deprecated and ignored; "
                "v1 scheduler always uses continuous mixed batching"
            )
            _SCHEDULE_HOMOGENEOUS_WARNED = True
    return False


def schedule_no_mixed_enabled() -> bool:
    """When true, v1 never mixes decode rows with prefill rows in one step."""
    return _truthy("INFINI_SCHEDULE_NO_MIXED", "0")


def schedule_preempt_enabled() -> bool:
    """When true, RECOMPUTE-preempt a RUNNING victim on decode append_slot OOM.

    Default on. Set ``INFINI_SCHEDULE_PREEMPT=0`` for defer-only behavior.
    """
    return _truthy("INFINI_SCHEDULE_PREEMPT", "1")


def long_prefill_threshold(default: int = 0) -> int:
    """Per-request prefill cap for one v1 step.

    Default 0 (inactive) when unset. Explicit ``INFINI_LONG_PREFILL_THRESHOLD``
    wins; otherwise chunked prefill uses ``max_num_batched_tokens()``.
    """
    raw = os.environ.get("INFINI_LONG_PREFILL_THRESHOLD")
    if raw:
        return int(raw)
    if prefill_chunked_enabled():
        return max_num_batched_tokens()
    return default


def moe_aot_step_max_tokens() -> int:
    """Largest MoE/piecewise AOT bucket (shape-gate ceiling)."""
    return _MOE_AOT_STEP_LADDER_MAX


def moe_aot_step_token_ladder() -> Tuple[int, ...]:
    """MoE/piecewise AOT bucket sizes (powers of two through 4096)."""
    return _MOE_AOT_STEP_TOKEN_LADDER


def moe_aot_step_total_allowed(total: int) -> bool:
    """True if step total can be served by MoE AOT (runtime pads up to a bucket)."""
    return 0 < total <= _MOE_AOT_STEP_LADDER_MAX
