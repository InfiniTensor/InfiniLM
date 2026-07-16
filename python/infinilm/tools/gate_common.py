# Copyright (c) 2025, InfiniCore
"""Re-export gate helpers; prefer ``scripts/moe_gate_lib`` to avoid package init."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from moe_gate_lib import (  # noqa: E402
    DEFAULT_BUCKETS,
    cache_root_default,
    gate_results_dir,
    model_path_default,
    workspace_root,
    write_gate_result,
)

__all__ = [
    "DEFAULT_BUCKETS",
    "cache_root_default",
    "gate_results_dir",
    "model_path_default",
    "workspace_root",
    "write_gate_result",
]
