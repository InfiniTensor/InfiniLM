#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""G4: verify MoE Triton cubin cache is populated; optional JIT-off dry-run."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _count_cache_entries(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    return sum(1 for _ in cache_dir.rglob("*") if _.is_file())


def main() -> int:
    from infinilm.compile.config import model_cache_hash
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
    )
    ap.add_argument(
        "--dry-run-launch",
        action="store_true",
        help="Launch pinned launcher once with JIT off (requires GPU + configs)",
    )
    args = ap.parse_args()
    buckets = sorted({int(x) for x in args.buckets.split(",") if x.strip()})
    cache_root = Path(args.cache_root).resolve()
    model_dir = cache_root / model_cache_hash(args.model_path)
    triton_cache = Path(
        os.environ.get("INFINI_MOE_TRITON_CACHE", str(model_dir / "moe_triton_cache"))
    )
    configs = Path(os.environ.get("INFINI_MOE_CONFIGS", str(model_dir / "moe_configs")))
    details: dict = {
        "triton_cache": str(triton_cache),
        "configs": str(configs),
        "buckets": buckets,
    }
    try:
        if not triton_cache.is_dir():
            raise RuntimeError(f"moe_triton_cache missing: {triton_cache}")
        n_files = _count_cache_entries(triton_cache)
        details["cache_file_count"] = n_files
        if n_files <= 0:
            raise RuntimeError(f"moe_triton_cache empty: {triton_cache}")
        if not configs.is_dir() or not any(configs.rglob("*.json")):
            raise RuntimeError(f"moe_configs missing/empty: {configs}")

        if args.dry_run_launch:
            import torch
            from infinilm.kernels.fused_moe_runtime import fused_moe_routed

            if not torch.cuda.is_available():
                raise RuntimeError("--dry-run-launch requires CUDA")
            os.environ["INFINI_MOE_CONFIGS"] = str(configs)
            os.environ["INFINI_MOE_TRITON_CACHE"] = str(triton_cache)
            os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
            os.environ["INFINI_MOE_ALLOW_JIT"] = "0"

            before = n_files
            H, E, N, K = 2048, 160, 512, 16
            device = torch.device("cuda", 0)
            dtype = torch.bfloat16
            dry_ms = [1, 16]
            for M in dry_ms:
                x = torch.randn(M, H, device=device, dtype=dtype)
                topk_ids = torch.randint(0, E, (M, K), device=device, dtype=torch.int32)
                topk_w = torch.softmax(
                    torch.randn(M, K, device=device, dtype=torch.float32), dim=-1
                ).to(dtype)
                w_gu = torch.randn(E, 2 * N, H, device=device, dtype=dtype).contiguous()
                w_d = torch.randn(E, H, N, device=device, dtype=dtype).contiguous()
                with torch.no_grad():
                    fused_moe_routed(x, topk_w, topk_ids, w_gu, w_d)
                torch.cuda.synchronize()
            after = _count_cache_entries(triton_cache)
            details["dry_run_M"] = dry_ms
            details["cache_files_before"] = before
            details["cache_files_after"] = after
            if after > before:
                raise RuntimeError(
                    f"Triton cache grew during JIT-disabled dry-run "
                    f"({before} → {after}); cubins incomplete for M={M}"
                )

        path = write_gate_result("G4", status="PASS", details=details)
        print(f"[G4] PASS cache_files={details['cache_file_count']} → {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        path = write_gate_result("G4", status="FAIL", details=details, error=str(exc))
        print(f"[G4] FAIL → {path}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
