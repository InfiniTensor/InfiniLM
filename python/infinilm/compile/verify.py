# Copyright (c) 2025, InfiniCore
"""Verify torch.compile graph split at flash-attn custom ops."""

from __future__ import annotations

import os
import re
from typing import Iterable, List


def iter_graph_sources(cache_dir: str) -> Iterable[str]:
    """Yield text sources under ``cache_dir`` that may contain FX graph dumps."""
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            if name == "computation_graph.py" or name.endswith(".py"):
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        yield f.read()
                except OSError:
                    continue


def count_splitting_ops_in_graph(text: str, splitting_ops: List[str]) -> dict[str, int]:
    """Count occurrences of each splitting op name in graph source text."""
    counts: dict[str, int] = {}
    for op in splitting_ops:
        base = op.split(".")[-1]
        patterns = [
            rf"\b{re.escape(op)}\b",
            rf"\b{re.escape(base)}\b",
            rf"infinilm\.{re.escape(base)}",
        ]
        hits = 0
        for pat in patterns:
            hits = max(hits, len(re.findall(pat, text)))
        counts[op] = hits
    return counts


def count_splitting_ops_in_cache(cache_dir: str, splitting_ops: List[str]) -> dict[str, int]:
    """Aggregate splitting-op hits across all graph-like files in the cache tree."""
    totals = {op: 0 for op in splitting_ops}
    for text in iter_graph_sources(cache_dir):
        partial = count_splitting_ops_in_graph(text, splitting_ops)
        for op, n in partial.items():
            totals[op] = max(totals[op], n)
    return totals


def list_triton_fused_artifacts(cache_dir: str) -> list[str]:
    """Return basenames of Triton fused kernel artifacts under ``triton_cache/``."""
    names: set[str] = set()
    for root, _dirs, files in os.walk(cache_dir):
        if "triton_cache" not in root:
            continue
        for name in files:
            if "fused" in name and name.startswith("triton_"):
                names.add(name.rsplit(".", 1)[0])
    return sorted(names)


def verify_flash_outside_compile(
    cache_dir: str,
    splitting_ops: List[str],
    *,
    min_op_hits: int = 1,
) -> tuple[bool, str]:
    """
    Check compile cache references splitting ops (flash outside Inductor).

    Returns:
        (ok, message)
    """
    counts = count_splitting_ops_in_cache(cache_dir, splitting_ops)
    total = sum(counts.values())
    if total >= min_op_hits:
        graph_path = None
        for root, _dirs, files in os.walk(cache_dir):
            if "computation_graph.py" in files:
                graph_path = os.path.join(root, "computation_graph.py")
                break
        return True, f"splitting ops in cache graphs: {counts} ({graph_path})"

    fused = list_triton_fused_artifacts(cache_dir)
    if fused:
        return (
            False,
            f"splitting op hits={counts}; triton fused artifacts present but "
            f"flash boundary not in graph yet: {fused[:5]}",
        )
    return False, f"splitting op hits too low: {counts} under {cache_dir}"
