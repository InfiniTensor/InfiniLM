# Copyright (c) 2025, InfiniCore
"""Environment flags for hybrid compiled prefill (Phases 3–6)."""

from __future__ import annotations

import os
from typing import List


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


def compile_max_seq_len(default: int = 8448) -> int:
    raw = os.environ.get("INFINI_COMPILE_MAX_SEQ")
    return int(raw) if raw else default


def compile_warmup_seq_lens(max_seq: int) -> List[int]:
    """Explicit Inductor warmup lengths (comma-separated ``COMPILE_WARMUP_SEQ_LENS``)."""
    raw = os.environ.get("COMPILE_WARMUP_SEQ_LENS")
    if raw:
        lens = [int(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        lens = [8, 64, min(512, max_seq)]
        for extra in (1024, 4096):
            if extra <= max_seq:
                lens.append(extra)
    if max_seq not in lens:
        lens.append(max_seq)
    return sorted({n for n in lens if 0 < n <= max_seq})
