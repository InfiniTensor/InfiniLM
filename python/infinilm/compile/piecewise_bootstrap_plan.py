# Copyright (c) 2025, InfiniCore
"""Shared package-collection plan for offline AOT compile and serve-time register.

Compile phase (``infinilm.server.entry --phase compile|all``) and InferEngine
bootstrap register/check must agree on the same ``segment.pt2`` set.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannedPackage:
    segment: str
    layer_idx: int
    bucket: int
    tp_rank: int
    package_path: str
    layer_agnostic: bool

    def exists(self) -> bool:
        return os.path.isfile(self.package_path)


@dataclass(frozen=True)
class PiecewiseBootstrapPlan:
    model_path: str
    model_type: str
    cache_root: str
    tp_size: int
    num_layers: int
    buckets: Tuple[int, ...]
    segments: Tuple[str, ...]
    layer_agnostic: bool
    packages: Tuple[PlannedPackage, ...]
    # MoE: buckets that must exist for strict validate (DEFAULT_BUCKETS subset).
    required_buckets: Tuple[int, ...]

    @property
    def expected_count(self) -> int:
        return len(self.packages)

    def missing_packages(self) -> List[PlannedPackage]:
        return [p for p in self.packages if not p.exists()]

    def full_cache_exists(self) -> bool:
        return not self.missing_packages()

    def missing_required(self) -> List[PlannedPackage]:
        """Packages that must exist for serve (all planned; MoE filters required buckets)."""
        if self.model_type != "minicpm5_moe" or not self.required_buckets:
            return self.missing_packages()
        req = set(self.required_buckets)
        return [
            p
            for p in self.packages
            if p.bucket in req and not p.exists()
        ]


def _load_hf_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def build_piecewise_bootstrap_plan(
    *,
    model_path: str,
    tp_size: int = 1,
    buckets: Optional[Sequence[int]] = None,
    cache_root: Optional[str] = None,
    num_layers: Optional[int] = None,
    layer_agnostic: Optional[bool] = None,
) -> PiecewiseBootstrapPlan:
    """Plan the full set of ``segment.pt2`` paths for compile and register."""
    from infinilm.compile.env import (
        compile_max_seq_len,
        native_piecewise_capture_buckets,
        piecewise_inductor_cache_root,
    )
    from infinilm.compile.piecewise_segments import (
        LAYER_AGNOSTIC_IDX,
        SEGMENT_PRE_ATTN,
        piecewise_inductor_package_path,
        piecewise_layer_agnostic_enabled,
    )
    from infinilm.compile.piecewise_moe_segment import SEGMENT_MOE

    hf_config = _load_hf_config(model_path)
    model_type = str(hf_config.get("model_type", ""))
    if num_layers is None:
        num_layers = int(hf_config.get("num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError(f"invalid num_hidden_layers for model_path={model_path}")

    tp_size = max(1, int(tp_size))
    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()
    root = cache_root or piecewise_inductor_cache_root()

    if buckets is None:
        buckets = native_piecewise_capture_buckets(compile_max_seq_len())

    required_buckets: Tuple[int, ...] = tuple()
    if model_type == "minicpm5_moe":
        from infinilm.tools.pack_moe_artifacts import DEFAULT_BUCKETS

        moe_ladder = tuple(DEFAULT_BUCKETS) + (8192,)
        buckets = tuple(sorted({int(b) for b in buckets} | set(moe_ladder)))
        segments: Tuple[str, ...] = (SEGMENT_MOE,)
        required_buckets = tuple(DEFAULT_BUCKETS)
    else:
        buckets = tuple(int(b) for b in buckets)
        segments = (SEGMENT_PRE_ATTN,)

    packages: List[PlannedPackage] = []
    layer_indices: Iterable[int]
    if layer_agnostic:
        layer_indices = (LAYER_AGNOSTIC_IDX,)
    else:
        layer_indices = range(num_layers)

    for segment in segments:
        for tp_rank in range(tp_size):
            for layer_idx in layer_indices:
                for bucket in buckets:
                    pkg = piecewise_inductor_package_path(
                        cache_root=root,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=int(layer_idx),
                        bucket=int(bucket),
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        layer_agnostic=layer_agnostic,
                        legacy_fallback=False,
                    )
                    packages.append(
                        PlannedPackage(
                            segment=segment,
                            layer_idx=int(layer_idx),
                            bucket=int(bucket),
                            tp_rank=tp_rank,
                            package_path=pkg,
                            layer_agnostic=bool(layer_agnostic),
                        )
                    )

    return PiecewiseBootstrapPlan(
        model_path=model_path,
        model_type=model_type,
        cache_root=root,
        tp_size=tp_size,
        num_layers=num_layers,
        buckets=tuple(buckets),
        segments=segments,
        layer_agnostic=bool(layer_agnostic),
        packages=tuple(packages),
        required_buckets=required_buckets,
    )


def compile_planned_packages(
    plan: PiecewiseBootstrapPlan,
    *,
    force: bool = False,
    tp_device_ids: Optional[Sequence[int]] = None,
    require_aot: bool = True,
) -> dict:
    """Offline AOT for missing planned packages (never inside a serving InferEngine)."""
    import torch

    from infinilm.compile.piecewise_segments import (
        SEGMENT_PRE_ATTN,
        aot_compile_piecewise_segments_multi_bucket,
    )
    from infinilm.compile.piecewise_moe_segment import (
        SEGMENT_MOE,
        aot_compile_minicpm5_moe_segment,
    )
    from infinilm.compile.env import piecewise_inductor_require_aot

    if tp_device_ids is None:
        tp_device_ids = list(range(plan.tp_size))
    require_aot = require_aot or piecewise_inductor_require_aot()

    if force:
        targets = list(plan.packages)
    else:
        targets = plan.missing_packages()

    if not targets:
        logger.info(
            "piecewise AOT compile: cache hit — all %s planned packages present under %s",
            plan.expected_count,
            plan.cache_root,
        )
        return {"compiled": 0, "skipped": plan.expected_count, "force": force}

    logger.info(
        "piecewise AOT compile: compiling %s/%s packages model_type=%s tp=%s "
        "buckets=%s cache=%s force=%s",
        len(targets),
        plan.expected_count,
        plan.model_type,
        plan.tp_size,
        list(plan.buckets),
        plan.cache_root,
        force,
    )

    compiled = 0
    try:
        for tp_rank in range(plan.tp_size):
            rank_targets = [p for p in targets if p.tp_rank == tp_rank]
            if not rank_targets:
                continue
            dev_index = (
                int(tp_device_ids[tp_rank])
                if tp_rank < len(tp_device_ids)
                else tp_rank
            )
            device = torch.device("cuda", dev_index)

            if SEGMENT_PRE_ATTN in plan.segments:
                pre_buckets = sorted(
                    {p.bucket for p in rank_targets if p.segment == SEGMENT_PRE_ATTN}
                )
                if pre_buckets:
                    summaries = aot_compile_piecewise_segments_multi_bucket(
                        model_path=plan.model_path,
                        segment=SEGMENT_PRE_ATTN,
                        buckets=pre_buckets,
                        device=device,
                        cache_root=plan.cache_root,
                        require_aot=require_aot,
                        skip_existing=not force,
                        tp_size=plan.tp_size,
                        tp_rank=tp_rank,
                        tp_device_ids=list(tp_device_ids),
                        layer_agnostic=plan.layer_agnostic,
                    )
                    compiled += len(summaries)

            if SEGMENT_MOE in plan.segments:
                moe_buckets = sorted(
                    {p.bucket for p in rank_targets if p.segment == SEGMENT_MOE}
                )
                for bucket in moe_buckets:
                    aot_compile_minicpm5_moe_segment(
                        model_path=plan.model_path,
                        bucket=bucket,
                        device=device,
                        cache_root=plan.cache_root,
                        tp_size=plan.tp_size,
                        tp_rank=tp_rank,
                        require_aot=require_aot,
                    )
                    compiled += 1
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    still_missing = plan.missing_packages() if not force else []
    if still_missing and require_aot:
        raise RuntimeError(
            "piecewise AOT compile incomplete: "
            f"{len(still_missing)} packages still missing; first={still_missing[0].package_path}"
        )

    logger.info(
        "piecewise AOT compile done: compiled=%s still_missing=%s",
        compiled,
        len(still_missing),
    )
    return {
        "compiled": compiled,
        "still_missing": len(still_missing),
        "force": force,
    }


def missing_packages_error_message(plan: PiecewiseBootstrapPlan, missing: Sequence[PlannedPackage]) -> str:
    first = missing[0].package_path if missing else "<none>"
    return (
        f"piecewise inductor packages missing: {len(missing)}/{plan.expected_count} "
        f"(SEGMENT=1, register-only). Compile offline first:\n"
        f"  python -m infinilm.server.entry --phase compile|all --model {plan.model_path} "
        f"--tp {plan.tp_size} ...\n"
        f"Or set INFINI_AOT_CHECK_SKIP=1 to skip register/check (debug only).\n"
        f"First missing: {first}"
    )
