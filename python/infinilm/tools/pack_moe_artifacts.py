#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Pack / verify MiniCPM5 MoE deploy artifacts (manifest + cross-refs)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def _model_hash_dir(cache_root: Path, model_path: str) -> Path:
    from infinilm.compile.config import model_cache_hash

    return cache_root / model_cache_hash(model_path)


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        pass
    return os.environ.get("INFINI_MOE_DEVICE_NAME", "X203")


def build_manifest(
    *,
    cache_root: Path,
    model_path: str,
    buckets: list[int],
    H: int = 2048,
    E: int = 160,
    N: int = 512,
) -> dict[str, Any]:
    from infinilm.kernels.moe_guards import launcher_hash

    model_dir = _model_hash_dir(cache_root, model_path)
    triton_ver = None
    try:
        import triton

        triton_ver = getattr(triton, "__version__", None)
    except Exception:
        pass

    return {
        "model_hash": model_dir.name,
        "model_path": os.path.abspath(model_path),
        "H": H,
        "E": E,
        "N": N,
        "device": _device_name(),
        "buckets": list(buckets),
        "launcher_hash": launcher_hash(),
        "triton_version": triton_ver,
        "configs_dir": "moe_configs",
        "triton_cache_dir": "moe_triton_cache",
    }


def verify_bundle(
    *,
    cache_root: Path,
    model_path: str,
    buckets: list[int] | None = None,
) -> dict[str, Any]:
    from infinilm.kernels.moe_guards import launcher_hash

    model_dir = _model_hash_dir(cache_root, model_path)
    manifest_path = model_dir / "moe_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = launcher_hash()
    if manifest.get("launcher_hash") != expected_hash:
        raise RuntimeError(
            f"launcher_hash mismatch: manifest={manifest.get('launcher_hash')} "
            f"current={expected_hash}. Re-run warmup + pack after launcher edits."
        )
    buckets = buckets or list(manifest.get("buckets") or DEFAULT_BUCKETS)
    missing_pt2: list[str] = []
    for b in buckets:
        # Layer-agnostic layout: .../tp1/rank0/moe_B{b}/segment.pt2
        candidates = list(model_dir.glob(f"**/moe_B{b}/segment.pt2"))
        if not any(p.is_file() for p in candidates):
            missing_pt2.append(f"moe_B{b}/segment.pt2")

    configs = model_dir / manifest.get("configs_dir", "moe_configs")
    triton_cache = model_dir / manifest.get("triton_cache_dir", "moe_triton_cache")
    if not configs.is_dir() or not any(configs.rglob("*.json")):
        raise RuntimeError(f"moe_configs missing or empty under {configs}")
    if not triton_cache.is_dir() or not any(triton_cache.iterdir()):
        raise RuntimeError(f"moe_triton_cache missing or empty: {triton_cache}")
    if missing_pt2:
        raise RuntimeError(
            "incomplete MoE AOT ladder (G3): missing " + ", ".join(missing_pt2)
        )
    return {
        "ok": True,
        "manifest": str(manifest_path),
        "buckets": buckets,
        "launcher_hash": expected_hash,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache-root",
        default=os.environ.get(
            "INFINI_PIECEWISE_INDUCTOR_CACHE",
            "bench_results/piecewise_inductor_cache_minicpm5",
        ),
    )
    ap.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "/models/minicpm5.16a3.v0314"),
    )
    ap.add_argument(
        "--buckets",
        default=",".join(str(b) for b in DEFAULT_BUCKETS),
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing manifest + artifacts (no write)",
    )
    args = ap.parse_args()
    cache_root = Path(args.cache_root).resolve()
    buckets = sorted({int(x) for x in args.buckets.split(",") if x.strip()})
    model_dir = _model_hash_dir(cache_root, args.model_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.verify:
        try:
            info = verify_bundle(
                cache_root=cache_root, model_path=args.model_path, buckets=buckets
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[pack-moe] VERIFY FAIL: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(info, indent=2))
        print("[pack-moe] VERIFY PASS")
        return 0

    # Ensure sibling dirs exist (may be populated by prior pipeline steps)
    (model_dir / "moe_configs").mkdir(parents=True, exist_ok=True)
    (model_dir / "moe_triton_cache").mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        cache_root=cache_root, model_path=args.model_path, buckets=buckets
    )
    dest = model_dir / "moe_manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[pack-moe] wrote {dest}")
    print(json.dumps(manifest, indent=2))

    try:
        verify_bundle(
            cache_root=cache_root, model_path=args.model_path, buckets=buckets
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[pack-moe] WARN: post-pack verify incomplete ({exc}). "
            "Run AOT + warmup before treating as G5 PASS.",
            file=sys.stderr,
        )
        return 0
    print("[pack-moe] VERIFY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
