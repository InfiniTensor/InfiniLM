#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""AOTInductor export for many piecewise segments with one weight load per rank batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

import torch


def _parse_layer_indices(spec: str, num_layers: int) -> list[int]:
    if spec.strip().lower() in ("all", "*", "canonical"):
        return list(range(num_layers))
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted({i for i in indices if 0 <= i < num_layers})


def _parse_buckets(spec: str) -> list[int]:
    spec_norm = spec.strip().lower()
    if spec_norm in ("all", "*"):
        from infinilm.compile.env import native_piecewise_capture_buckets, compile_max_seq_len

        return list(native_piecewise_capture_buckets(compile_max_seq_len()))
    if spec_norm in ("sub512",):
        from infinilm.compile.piecewise_segments import sub512_farm_buckets

        return list(sub512_farm_buckets())
    if spec_norm in ("vllm-ladder", "vllm_ladder"):
        from infinilm.compile.piecewise_segments import vllm_aligned_farm_buckets

        return list(vllm_aligned_farm_buckets())
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment", default="pre_attn", choices=("pre_attn", "post_attn_cg"))
    parser.add_argument("--bucket", type=int, default=0, help="Single bucket (omit when --buckets set)")
    parser.add_argument(
        "--buckets",
        default="",
        help="Comma-separated buckets, 'all' (native capture ladder), or 'sub512' (1..512 powers)",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--layers", default="all", help="Layer indices: all, canonical, 0-31, or 0,1,5")
    parser.add_argument(
        "--valid-seq-len",
        type=int,
        default=0,
        help="Valid tokens (0 = full bucket)",
    )
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--require-aot", action="store_true")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompile even when segment.pt2 already exists",
    )
    parser.add_argument(
        "--layer-agnostic",
        action="store_true",
        help="Export one layer-agnostic package per bucket (external weights at runtime)",
    )
    parser.add_argument(
        "--legacy-per-layer",
        action="store_true",
        help="Force legacy per-layer artifact paths (overrides INFINI_PIECEWISE_LAYER_AGNOSTIC)",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from infinilm.compile.env import (
        piecewise_inductor_cache_root,
        piecewise_inductor_require_aot,
    )
    from infinilm.compile.piecewise_segments import (
        aot_compile_piecewise_segments_batch,
        aot_compile_piecewise_segments_multi_bucket,
        expected_piecewise_package_count,
        piecewise_layer_agnostic_enabled,
    )

    if args.legacy_per_layer:
        os.environ["INFINI_PIECEWISE_LAYER_AGNOSTIC"] = "0"
    elif args.layer_agnostic:
        os.environ["INFINI_PIECEWISE_LAYER_AGNOSTIC"] = "1"

    layer_agnostic = piecewise_layer_agnostic_enabled()

    with open(os.path.join(args.model_path, "config.json"), encoding="utf-8") as f:
        num_layers = int(json.load(f)["num_hidden_layers"])

    device = torch.device(args.device)
    cache_root = args.cache_root or piecewise_inductor_cache_root()
    require_aot = args.require_aot or piecewise_inductor_require_aot()
    layer_indices = _parse_layer_indices(args.layers, num_layers)
    tp_size = max(1, int(args.tp_size))
    tp_rank = int(args.tp_rank)
    if tp_size > 1:
        tp_device_ids = list(range(tp_size))
    elif device.type == "cuda" and device.index is not None:
        tp_device_ids = [int(device.index)]
    else:
        tp_device_ids = [0]

    buckets = _parse_buckets(args.buckets) if args.buckets else [int(args.bucket)]
    if not buckets:
        print("[aot_segments] FAIL: specify --bucket or --buckets", flush=True)
        return 1

    print(
        f"[aot_segments] segment={args.segment} buckets={buckets} "
        f"tp_size={tp_size} tp_rank={tp_rank} layer_agnostic={layer_agnostic} "
        f"layers={layer_indices[0]}-{layer_indices[-1]} ({len(layer_indices)} total) "
        f"model={args.model_path} require_aot={require_aot}",
        flush=True,
    )

    try:
        if len(buckets) > 1 or args.buckets:
            summaries = aot_compile_piecewise_segments_multi_bucket(
                model_path=args.model_path,
                segment=args.segment,
                buckets=buckets,
                device=device,
                cache_root=cache_root,
                layer_indices=layer_indices,
                valid_seq_len=args.valid_seq_len if args.valid_seq_len > 0 else None,
                require_aot=require_aot,
                skip_existing=not args.no_skip_existing,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_device_ids=tp_device_ids,
                layer_agnostic=layer_agnostic,
            )
        else:
            bucket = buckets[0]
            valid_seq_len = args.valid_seq_len if args.valid_seq_len > 0 else bucket
            summaries = aot_compile_piecewise_segments_batch(
                model_path=args.model_path,
                segment=args.segment,
                layer_indices=layer_indices,
                bucket=bucket,
                device=device,
                cache_root=cache_root,
                valid_seq_len=valid_seq_len,
                require_aot=require_aot,
                skip_existing=not args.no_skip_existing,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_device_ids=tp_device_ids,
                layer_agnostic=layer_agnostic,
            )
    except RuntimeError as exc:
        print(f"[aot_segments] FAIL: {exc}", flush=True)
        return 1

    if require_aot:
        for summary in summaries:
            if summary.get("backend") != "aot_inductor":
                print(
                    f"[aot_segments] FAIL require_aot set but backend={summary.get('backend')}",
                    flush=True,
                )
                return 1

    expected = expected_piecewise_package_count(
        num_layers=num_layers,
        buckets=buckets,
        tp_size=tp_size,
        layer_agnostic=layer_agnostic,
    )
    print(
        f"[aot_segments] PASS compiled={len(summaries)} buckets={buckets} "
        f"expected_pkgs_per_rank={expected // tp_size if tp_size else expected}",
        flush=True,
    )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summaries": summaries,
                    "buckets": buckets,
                    "layer_agnostic": layer_agnostic,
                    "expected_packages": expected,
                },
                f,
                indent=2,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
