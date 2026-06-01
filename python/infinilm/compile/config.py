# Copyright (c) 2025, InfiniCore
"""Compilation config aligned with vLLM 0.10.2 ``CompilationConfig``."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .env import (
    compile_buckets,
    prefill_cudagraph_capture_buckets,
    prefill_cudagraph_enabled,
    prefill_cudagraph_max_bucket,
)

# Flash-attn custom op kept outside Inductor (vLLM ``splitting_ops`` pattern).
DEFAULT_SPLITTING_OPS: List[str] = ["infinilm.prefill_flash_attention"]


def default_compile_size_ladder(max_seq_len: int = 8192) -> List[int]:
    """Powers-of-two ladder up to ``max_seq_len`` (vLLM-style capture/compile sizes)."""
    sizes: List[int] = []
    s = 1
    while s < max_seq_len:
        sizes.append(s)
        s *= 2
    sizes.append(max_seq_len)
    return sizes


def model_cache_hash(model_path: str) -> str:
    """Stable id from ``config.json`` (weights path basename + config hash)."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    base = os.path.basename(os.path.normpath(model_path))
    return f"{base}_{h}"


@dataclass
class CompiledPrefillConfig:
    """InfiniLM-owned torch.compile cache for the prefill backbone."""

    model_path: str
    max_seq_len: int = 8448
    compile_sizes: Optional[List[int]] = None
    cudagraph_capture_sizes: Optional[List[int]] = None
    cache_root: str = field(
        default_factory=lambda: os.environ.get(
            "INFINI_TORCH_COMPILE_CACHE",
            "bench_results/torch_compile_cache",
        )
    )
    rank: int = 0
    prefix: str = "prefill_backbone"
    # Phase 4: vLLM piecewise CUDAGraph when ``INFINI_PREFILL_CUDAGRAPH=1``.
    use_cudagraph: bool = field(default_factory=prefill_cudagraph_enabled)
    splitting_ops: Optional[List[str]] = None
    enable_fusion: bool = True

    def __post_init__(self) -> None:
        if self.compile_sizes is None:
            self.compile_sizes = default_compile_size_ladder(self.max_seq_len)
        self.compile_sizes = sorted(
            set(self.compile_sizes) | set(compile_buckets(self.max_seq_len))
        )
        if self.cudagraph_capture_sizes is None:
            if self.use_cudagraph:
                explicit = prefill_cudagraph_capture_buckets(self.max_seq_len)
                if explicit is not None:
                    self.cudagraph_capture_sizes = list(explicit)
                else:
                    buckets = list(compile_buckets(self.max_seq_len))
                    max_cg = prefill_cudagraph_max_bucket()
                    if max_cg is not None:
                        buckets = [b for b in buckets if b <= max_cg]
                    self.cudagraph_capture_sizes = buckets
            else:
                self.cudagraph_capture_sizes = list(self.compile_sizes)
        if self.splitting_ops is None:
            self.splitting_ops = list(DEFAULT_SPLITTING_OPS)

    @property
    def model_hash(self) -> str:
        return model_cache_hash(self.model_path)

    @property
    def cache_dir(self) -> str:
        return os.path.join(
            self.cache_root,
            self.model_hash,
            f"rank_{self.rank}",
            self.prefix,
        )

    def write_metadata(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        meta = {
            "model_path": self.model_path,
            "max_seq_len": self.max_seq_len,
            "compile_sizes": self.compile_sizes,
            "cudagraph_capture_sizes": self.cudagraph_capture_sizes,
            "splitting_ops": self.splitting_ops,
            "enable_fusion": self.enable_fusion,
        }
        with open(os.path.join(self.cache_dir, "infinilm_compile_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
