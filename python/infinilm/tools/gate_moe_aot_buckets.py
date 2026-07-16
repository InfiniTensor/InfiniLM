#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""G3: verify every MoE ladder bucket has segment.pt2 on disk."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    from infinilm.compile.config import model_cache_hash
    from infinilm.compile.piecewise_moe_segment import SEGMENT_MOE
    from infinilm.compile.piecewise_segments import (
        LAYER_AGNOSTIC_IDX,
        piecewise_inductor_package_path,
    )
    from infinilm.tools.gate_common import (
        DEFAULT_BUCKETS,
        cache_root_default,
        model_path_default,
        write_gate_result,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", default=str(cache_root_default()))
    ap.add_argument("--model-path", default=model_path_default())
    ap.add_argument(
        "--buckets",
        default=",".join(str(b) for b in DEFAULT_BUCKETS),
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify on-disk packages (default)",
    )
    args = ap.parse_args()
    buckets = sorted({int(x) for x in args.buckets.split(",") if x.strip()})
    cache_root = Path(args.cache_root).resolve()
    details: dict = {
        "cache_root": str(cache_root),
        "model_path": args.model_path,
        "model_hash": model_cache_hash(args.model_path),
        "buckets": buckets,
    }
    try:
        missing: list[str] = []
        present: list[str] = []
        for b in buckets:
            pkg = piecewise_inductor_package_path(
                cache_root=str(cache_root),
                model_path=args.model_path,
                segment=SEGMENT_MOE,
                layer_idx=LAYER_AGNOSTIC_IDX,
                bucket=int(b),
                tp_size=1,
                tp_rank=0,
                layer_agnostic=True,
            )
            if os.path.isfile(pkg):
                present.append(pkg)
            else:
                missing.append(f"moe_B{b}/segment.pt2 ({pkg})")
        details["present_count"] = len(present)
        details["expected_count"] = len(buckets)
        details["missing"] = missing
        if missing:
            raise RuntimeError(
                f"incomplete MoE AOT ladder: missing {len(missing)}/{len(buckets)}: "
                + "; ".join(missing[:5])
            )
        if details["present_count"] != details["expected_count"]:
            raise RuntimeError(
                f"registered/present mismatch: present={details['present_count']} "
                f"expected={details['expected_count']}"
            )
        path = write_gate_result("G3", status="PASS", details=details)
        print(f"[G3] PASS present={len(present)} → {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        path = write_gate_result("G3", status="FAIL", details=details, error=str(exc))
        print(f"[G3] FAIL → {path}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
