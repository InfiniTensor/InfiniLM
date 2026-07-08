# Copyright (c) 2025, InfiniCore
"""PRD-04 torch.compile config (vLLM-free)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from .env import (
    compile_buckets,
    compile_max_seq_len,
    torch_compile_cache_root,
    vllm_unified_power_ladder,
)

# Flash-attn custom op kept outside Inductor (enforced via splitting_flash_boundary at load).
DEFAULT_SPLITTING_OPS: List[str] = ["infinilm.prefill_flash_attention"]


def default_compile_size_ladder(max_seq_len: int = 8192) -> List[int]:
    """Unified power-of-2 ladder (+ ``max_seq_len`` tail when > 8192)."""
    return list(vllm_unified_power_ladder(max_seq_len))


def model_cache_hash(model_path: str) -> str:
    """Stable id from ``config.json`` (weights path basename + config hash)."""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()
    h = hashlib.sha256(raw.encode()).hexdigest()[:12]
    base = os.path.basename(os.path.normpath(model_path))
    return f"{base}_{h}"


@dataclass
class TorchCompileConfig:
    """InfiniLM-owned torch.compile cache for the unified prefill backbone."""

    model_path: str
    max_seq_len: int = field(default_factory=compile_max_seq_len)
    compile_sizes: Optional[List[int]] = None
    cache_root: str = field(default_factory=torch_compile_cache_root)
    rank: int = 0
    prefix: str = "torch_compile"
    splitting_ops: Optional[List[str]] = None
    enable_fusion: bool = True

    def __post_init__(self) -> None:
        if self.compile_sizes is None:
            self.compile_sizes = default_compile_size_ladder(self.max_seq_len)
        self.compile_sizes = sorted(
            set(self.compile_sizes) | set(compile_buckets(self.max_seq_len))
        )
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
            "splitting_ops": self.splitting_ops,
            "enable_fusion": self.enable_fusion,
        }
        with open(os.path.join(self.cache_dir, "infinilm_compile_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
