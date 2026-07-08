#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""M4 Phase 1.5: AOTInductor export for one native piecewise segment."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional, Sequence

import torch


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment", default="pre_attn", choices=("pre_attn", "post_attn_cg"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--bucket", type=int, default=512)
    parser.add_argument(
        "--valid-seq-len",
        type=int,
        default=0,
        help="Valid tokens (0 = full bucket)",
    )
    parser.add_argument("--cache-root", default="")
    parser.add_argument(
        "--require-aot",
        action="store_true",
        help="Fail if AOT packaging fails (no torch.compile fallback)",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from infinilm.compile.env import (
        piecewise_inductor_cache_root,
        piecewise_inductor_require_aot,
    )
    from infinilm.compile.piecewise_segments import aot_compile_piecewise_segment

    device = torch.device(args.device)
    cache_root = args.cache_root or piecewise_inductor_cache_root()
    valid_seq_len = args.valid_seq_len if args.valid_seq_len > 0 else args.bucket
    require_aot = args.require_aot or piecewise_inductor_require_aot()

    print(
        f"[aot_segment] segment={args.segment} layer={args.layer} "
        f"bucket={args.bucket} valid={valid_seq_len} model={args.model_path} "
        f"require_aot={require_aot}",
        flush=True,
    )

    try:
        t0 = time.perf_counter()
        summary = aot_compile_piecewise_segment(
            model_path=args.model_path,
            segment=args.segment,
            layer_idx=args.layer,
            bucket=args.bucket,
            device=device,
            cache_root=cache_root,
            valid_seq_len=valid_seq_len,
            require_aot=require_aot,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        summary["total_ms"] = elapsed_ms
    except NotImplementedError as exc:
        print(f"[aot_segment] SKIP: {exc}", flush=True)
        return 2
    except RuntimeError as exc:
        print(f"[aot_segment] FAIL: {exc}", flush=True)
        return 1

    backend = summary.get("backend", "")
    if require_aot and backend != "aot_inductor":
        print(
            f"[aot_segment] FAIL require_aot set but backend={backend}",
            flush=True,
        )
        return 1

    print(
        f"[aot_segment] PASS backend={backend} package={summary.get('package_path', '')} "
        f"total_ms={summary.get('total_ms', 0):.1f}",
        flush=True,
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
