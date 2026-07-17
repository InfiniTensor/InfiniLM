#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""Parity gate: aten capture-safe MoE vs Triton cubins (bf16).

Compares ``_routed_experts_aten`` (CG-2 body under ``hcStreamBeginCapture``) to
Triton ``fused_moe_routed`` for M∈{1,16}, TOP_K=16. Fail-closed before any serve
flip of ``INFINI_MOE_CAPTURE_SAFE=1``.

Pass: bf16 max abs < 1e-2 (same spirit as M5 G2).

Artifact (optional)::

    profiling/hctracer_fusedmoe/<ts>/parity_capture_safe/

Env::

    INFINI_MOE_CONFIGS, INFINI_MOE_TRITON_CACHE (or TRITON_CACHE_DIR)
    PARITY_MS          default 1,16
    PARITY_OUT_DIR     optional JSON/log directory
    INFINI_MOE_ALLOW_JIT  default 0 for serve-faithful cubin path; set 1 to allow JIT
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for p in (
    _ROOT / "InfiniLM" / "python",
    _ROOT / "InfiniCore" / "python",
    _ROOT / "scripts",
):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _write_gate(status: str, details: dict[str, Any], error: str | None = None) -> Path:
    from infinilm.tools.gate_common import write_gate_result

    return write_gate_result("parity_capture_safe", status=status, details=details, error=error)


def _write_artifact(out_dir: Path, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "result.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> int:
    Ms = [int(x) for x in os.environ.get("PARITY_MS", "1,16").split(",") if x]
    H, E, N, TOP_K = 2048, 160, 512, 16
    threshold = 1e-2
    details: dict[str, Any] = {
        "Ms": Ms,
        "TOP_K": TOP_K,
        "H": H,
        "E": E,
        "N": N,
        "threshold": threshold,
        "dtype": "bfloat16",
    }

    out_dir_env = os.environ.get("PARITY_OUT_DIR", "").strip()
    if out_dir_env:
        out_dir = Path(out_dir_env)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = (
            _ROOT / "profiling" / "hctracer_fusedmoe" / ts / "parity_capture_safe"
        )

    if not os.environ.get("INFINI_MOE_CONFIGS"):
        print("FATAL: set INFINI_MOE_CONFIGS", file=sys.stderr)
        path = _write_gate("FAIL", details, "INFINI_MOE_CONFIGS unset")
        _write_artifact(out_dir, {"status": "FAIL", "error": "INFINI_MOE_CONFIGS unset", **details})
        print(f"[parity_capture_safe] FAIL → {path}", file=sys.stderr)
        return 2
    if not (
        os.environ.get("INFINI_MOE_TRITON_CACHE") or os.environ.get("TRITON_CACHE_DIR")
    ):
        print("FATAL: set INFINI_MOE_TRITON_CACHE", file=sys.stderr)
        path = _write_gate("FAIL", details, "INFINI_MOE_TRITON_CACHE unset")
        _write_artifact(
            out_dir, {"status": "FAIL", "error": "INFINI_MOE_TRITON_CACHE unset", **details}
        )
        print(f"[parity_capture_safe] FAIL → {path}", file=sys.stderr)
        return 2

    # Serve-faithful default: cubins only. Recompile containers may set ALLOW_JIT=1.
    os.environ.setdefault("INFINI_MOE_ALLOW_JIT", "0")
    # Ensure Triton path (not aten) for the reference side.
    os.environ.pop("INFINI_MOE_CAPTURE_SAFE", None)

    try:
        import torch
        from infinilm.kernels.fused_moe_runtime import (
            _routed_experts_aten,
            fused_moe_routed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("parity_capture_safe requires CUDA/PrivateUse1")

        device = torch.device("cuda", 0)
        dtype = torch.bfloat16
        scale = float(H) ** -0.5
        details["weight_scale"] = scale

        per_m: dict[str, float] = {}
        for M in Ms:
            torch.manual_seed(M)
            x = (torch.randn(M, H, device=device, dtype=dtype) * scale).contiguous()
            topk_ids = torch.randint(0, E, (M, TOP_K), device=device, dtype=torch.int32)
            topk_w = torch.softmax(
                torch.randn(M, TOP_K, device=device, dtype=torch.float32), dim=-1
            ).to(dtype)
            w_gate_up = (
                torch.randn(E, 2 * N, H, device=device, dtype=dtype) * scale
            ).contiguous()
            w_down = (
                torch.randn(E, H, N, device=device, dtype=dtype) * scale
            ).contiguous()

            with torch.no_grad():
                out_aten = _routed_experts_aten(x, topk_w, topk_ids, w_gate_up, w_down)
                out_triton = fused_moe_routed(x, topk_w, topk_ids, w_gate_up, w_down)
            diff = (out_aten.float() - out_triton.float()).abs().max().item()
            per_m[str(M)] = diff
            print(f"[parity_capture_safe] M={M} max_abs={diff:.6g}", flush=True)
            if diff >= threshold:
                raise RuntimeError(
                    f"aten vs Triton parity fail M={M} max_abs={diff} (>= {threshold})"
                )

        details["max_abs_per_M"] = per_m
        details["out_dir"] = str(out_dir)
        path = _write_gate("PASS", details)
        art = _write_artifact(
            out_dir,
            {
                "status": "PASS",
                "gate": "parity_capture_safe",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **details,
            },
        )
        print(f"[parity_capture_safe] PASS → {path}")
        print(f"[parity_capture_safe] artifact → {art}")
        return 0
    except Exception as exc:  # noqa: BLE001
        details["out_dir"] = str(out_dir)
        path = _write_gate("FAIL", details, str(exc))
        _write_artifact(
            out_dir,
            {
                "status": "FAIL",
                "gate": "parity_capture_safe",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **details,
            },
        )
        print(f"[parity_capture_safe] FAIL → {path}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
