#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Warm Triton cubin cache for the pinned InfiniLM FusedMoE launcher.

Runs the Tier-2 launcher (not serve-time vLLM imports) for each MoE bucket so
``TRITON_CACHE_DIR`` / ``INFINI_MOE_TRITON_CACHE`` contain cubins keyed to
``fused_moe_runtime.py``. Optional vLLM parity when ``--parity`` and vLLM are
available (recompile container only).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
H, E, N, TOP_K = 2048, 160, 512, 8


def _parse_buckets(raw: str) -> list[int]:
    return sorted({int(x) for x in raw.split(",") if x.strip()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--buckets",
        default=",".join(str(b) for b in DEFAULT_BUCKETS),
        help="Comma-separated token counts M (decode/prefill ladders)",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument(
        "--parity",
        action="store_true",
        help="Compare launcher vs vLLM fused_experts when importable",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Warmup may JIT into the cache.
    os.environ["INFINI_MOE_ALLOW_JIT"] = "1"
    if not os.environ.get("INFINI_MOE_TRITON_CACHE") and not os.environ.get(
        "TRITON_CACHE_DIR"
    ):
        print(
            "FATAL: set INFINI_MOE_TRITON_CACHE (or TRITON_CACHE_DIR)",
            file=sys.stderr,
        )
        return 2
    if not os.environ.get("INFINI_MOE_CONFIGS"):
        print("FATAL: set INFINI_MOE_CONFIGS to moe_configs/", file=sys.stderr)
        return 2

    cache = Path(
        os.environ.get("INFINI_MOE_TRITON_CACHE")
        or os.environ["TRITON_CACHE_DIR"]
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache)

    import torch
    from infinilm.kernels.fused_moe_runtime import fused_moe_routed, launcher_hash

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    device = torch.device("cuda", 0)
    torch.manual_seed(args.seed)
    buckets = _parse_buckets(args.buckets)
    print(f"[warmup-moe] launcher_hash={launcher_hash()}")
    print(f"[warmup-moe] cache={cache}")
    print(f"[warmup-moe] configs={os.environ['INFINI_MOE_CONFIGS']}")
    print(f"[warmup-moe] buckets={buckets}")

    for M in buckets:
        x = torch.randn(M, H, device=device, dtype=dtype)
        topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
        topk_w = torch.softmax(
            torch.randn(M, TOP_K, device=device, dtype=torch.float32), dim=-1
        ).to(dtype)
        w_gate_up = torch.randn(E, 2 * N, H, device=device, dtype=dtype)
        w_down = torch.randn(E, H, N, device=device, dtype=dtype)
        # Contiguous last dim for Triton stride contract
        w_gate_up = w_gate_up.contiguous()
        w_down = w_down.contiguous()
        with torch.no_grad():
            out = fused_moe_routed(x, topk_w, topk_ids, w_gate_up, w_down)
        torch.cuda.synchronize()
        print(
            f"[warmup-moe] M={M} out_norm={float(out.float().norm()):.4f}",
            flush=True,
        )

        if args.parity:
            # moe_sum vs reference (torch / vLLM when present)
            cache3 = torch.randn(M, TOP_K, H, device=device, dtype=dtype)
            from infinilm.kernels.fused_moe_runtime import moe_sum as _moe_sum

            ours = _moe_sum(cache3)
            ref_sum = cache3.float().sum(dim=1).to(dtype)
            sum_diff = (ours.float() - ref_sum.float()).abs().max().item()
            print(f"[warmup-moe] M={M} moe_sum_vs_torch={sum_diff:.6g}", flush=True)
            if sum_diff >= 1e-2:
                print(
                    f"FATAL: moe_sum parity fail M={M} max_abs={sum_diff}",
                    file=sys.stderr,
                )
                return 1
            try:
                from vllm import _custom_ops as vllm_ops

                vllm_out = torch.empty(M, H, device=device, dtype=dtype)
                vllm_ops.moe_sum(cache3, vllm_out)
                vdiff = (ours.float() - vllm_out.float()).abs().max().item()
                print(f"[warmup-moe] M={M} moe_sum_vs_vllm={vdiff:.6g}", flush=True)
                if vdiff >= 1e-2:
                    print(
                        f"FATAL: moe_sum vs vLLM fail M={M} max_abs={vdiff}",
                        file=sys.stderr,
                    )
                    return 1
            except Exception as exc:  # noqa: BLE001
                print(f"[warmup-moe] moe_sum vLLM parity skipped: {exc}")

            try:
                from vllm.model_executor.layers.fused_moe import fused_experts
            except Exception as exc:  # noqa: BLE001
                print(f"[warmup-moe] fused_experts parity skipped (no vLLM): {exc}")
                continue
            with torch.no_grad():
                ref = fused_experts(
                    x,
                    w_gate_up,
                    w_down,
                    topk_w,
                    topk_ids,
                    inplace=False,
                )
            diff = (out.float() - ref.float()).abs().max().item()
            print(f"[warmup-moe] M={M} max_abs_vs_vllm={diff:.6g}", flush=True)
            if diff >= 1e-2:
                print(
                    f"FATAL: parity fail M={M} max_abs={diff} (>=1e-2)",
                    file=sys.stderr,
                )
                return 1

    print("[warmup-moe] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
