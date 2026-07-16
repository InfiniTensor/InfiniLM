#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Minimal FusedMoE microbench for hcTracer (no vLLM, no serve).

Modes:
  launcher  — fused_moe_routed only for M in {1,16}, TOP_K=16
  aoti_b16  — aoti_load_package(moe_B16/segment.pt2) with hidden [1,16,H]

Env (required; wrapper sets defaults):
  INFINI_MOE_CONFIGS, INFINI_MOE_TRITON_CACHE / TRITON_CACHE_DIR,
  INFINI_MOE_ALLOW_JIT=0

Refuses if Triton cache grows during the run (G4 spirit).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


H, E, N, TOP_K = 2048, 160, 512, 16
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


def _count_cache_entries(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    return sum(1 for _ in cache_dir.rglob("*") if _.is_file())


def _cache_dir() -> Path:
    raw = (
        os.environ.get("INFINI_MOE_TRITON_CACHE", "").strip()
        or os.environ.get("TRITON_CACHE_DIR", "").strip()
    )
    if not raw:
        raise RuntimeError("INFINI_MOE_TRITON_CACHE / TRITON_CACHE_DIR unset")
    return Path(raw)


def _require_jit_off() -> None:
    if os.environ.get("INFINI_MOE_ALLOW_JIT", "0").strip() not in ("0", ""):
        raise RuntimeError(
            "INFINI_MOE_ALLOW_JIT must be 0 for profiling (got "
            f"{os.environ.get('INFINI_MOE_ALLOW_JIT')!r})"
        )
    os.environ["INFINI_MOE_ALLOW_JIT"] = "0"


def _refuse_vllm() -> None:
    if "vllm" in sys.modules:
        raise RuntimeError("vllm already imported; refuse (serve-free microbench)")


def _make_launcher_inputs(M: int, device, dtype):
    import torch

    x = torch.randn(M, H, device=device, dtype=dtype)
    topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
    topk_w = torch.softmax(
        torch.randn(M, TOP_K, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)
    w_gu = torch.randn(E, 2 * N, H, device=device, dtype=dtype).contiguous()
    w_d = torch.randn(E, H, N, device=device, dtype=dtype).contiguous()
    return x, topk_w, topk_ids, w_gu, w_d


def _run_timed(label: str, fn, *, warmup: int, iters: int, device) -> float:
    import torch

    print(f"=== WARMUP_BEGIN mode={label} warmup={warmup} ===", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    print(f"=== WARMUP_END mode={label} ===", flush=True)

    print(f"=== TIMED_BEGIN mode={label} iters={iters} ===", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_per_iter = elapsed_ms / max(iters, 1)
    print(
        f"=== TIMED_END mode={label} host_ms_total={elapsed_ms:.3f} "
        f"host_ms_per_iter={ms_per_iter:.3f} iters={iters} ===",
        flush=True,
    )
    print(f"[profile-moe] {label} host_ms/iter={ms_per_iter:.3f}", flush=True)
    return ms_per_iter


def _run_launcher(args) -> int:
    import torch
    from infinilm.kernels.fused_moe_runtime import fused_moe_routed, launcher_hash

    _refuse_vllm()
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    M = int(args.M)
    if M not in (1, 16):
        raise ValueError(f"launcher mode expects M in {{1,16}}, got {M}")

    print(f"[profile-moe] mode=launcher M={M} TOP_K={TOP_K} H={H} E={E} N={N}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(f"[profile-moe] configs={os.environ.get('INFINI_MOE_CONFIGS')}")
    print(f"[profile-moe] triton_cache={_cache_dir()}")

    x, topk_w, topk_ids, w_gu, w_d = _make_launcher_inputs(M, device, dtype)

    def once():
        return fused_moe_routed(x, topk_w, topk_ids, w_gu, w_d)

    _run_timed(
        f"launcher_m{M}",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    return 0


def _run_aoti_b16(args) -> int:
    import torch
    from torch._inductor import aoti_load_package

    from infinilm.compile.piecewise_moe_segment import make_moe_example_inputs
    from infinilm.kernels.fused_moe_runtime import launcher_hash

    _refuse_vllm()
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    pkg = Path(args.segment_pt2).resolve()
    if not pkg.is_file():
        raise FileNotFoundError(f"moe_B16 segment.pt2 missing: {pkg}")

    print(f"[profile-moe] mode=aoti_b16 package={pkg}")
    print(f"[profile-moe] launcher_hash={launcher_hash()}")
    print(
        f"[profile-moe] shapes hidden=[1,16,{H}] (bucket pad; valid baked in pt2) "
        f"TOP_K={TOP_K} (segment routing)"
    )

    # Ensure opaque op registration before load/run.
    from infinilm.torch_llama.moe_ops import register_fused_moe_routed_op

    register_fused_moe_routed_op()

    device_index = device.index if device.index is not None else 0
    runner = aoti_load_package(str(pkg), device_index=device_index)
    inputs = make_moe_example_inputs(
        bucket=16,
        hidden_size=H,
        moe_intermediate_size=N,
        n_routed_experts=E,
        device=device,
        dtype=dtype,
    )

    def once():
        return runner(*inputs)

    _run_timed(
        "aoti_b16",
        once,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        required=True,
        choices=("launcher", "aoti_b16"),
    )
    ap.add_argument("--M", type=int, default=1, help="launcher token count (1 or 16)")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument(
        "--out-dir",
        default="",
        help="Optional dir for a small stdout copy / marker (profiling tree)",
    )
    ap.add_argument(
        "--segment-pt2",
        default="",
        help="Path to moe_B16/segment.pt2 (aoti_b16); default under deploy cache",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        _require_jit_off()
        if not os.environ.get("INFINI_MOE_CONFIGS", "").strip():
            raise RuntimeError("INFINI_MOE_CONFIGS unset")
        cache = _cache_dir()
        before = _count_cache_entries(cache)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.manual_seed(args.seed)

        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)

        if args.mode == "launcher":
            rc = _run_launcher(args)
        else:
            if not args.segment_pt2:
                raise RuntimeError("--segment-pt2 required for aoti_b16 (or set via wrapper)")
            rc = _run_aoti_b16(args)

        after = _count_cache_entries(cache)
        print(
            f"[profile-moe] cache_files before={before} after={after}",
            flush=True,
        )
        if after > before:
            raise RuntimeError(
                f"Triton cache grew during JIT-off profile ({before} → {after}); "
                "cubins incomplete for this shape/TOP_K"
            )
        return rc
    except Exception as exc:  # noqa: BLE001
        print(f"[profile-moe] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
