#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""DEPRECATED (Phase 1): aten capture-safe MoE parity gate.

``_routed_experts_aten`` / ``INFINI_MOE_CAPTURE_SAFE`` were removed. This tool
now hard-fails closed so CI / scripts cannot silently flip to a deleted body.

Jul21 / Phase 0–1 north star: Triton ``fused_moe_routed`` under capture.
Use ``inductor_moe_hcgraph_smoke.py --mode triton_capture`` instead.
"""
from __future__ import annotations

import sys
from typing import Any


def _write_gate(status: str, details: dict[str, Any], error: str | None = None):
    from infinilm.tools.gate_common import write_gate_result

    return write_gate_result(
        "parity_capture_safe", status=status, details=details, error=error
    )


def main() -> int:
    msg = (
        "parity_capture_safe_moe deprecated (Phase 1): aten MoE capture escape "
        "(_routed_experts_aten / INFINI_MOE_CAPTURE_SAFE) deleted. Use "
        "inductor_moe_hcgraph_smoke.py --mode triton_capture."
    )
    details = {"deprecated": True, "replacement": "triton_capture smoke"}
    try:
        path = _write_gate("FAIL", details, msg)
        print(f"[parity_capture_safe] FAIL → {path}: {msg}", file=sys.stderr)
    except Exception:  # noqa: BLE001
        print(f"[parity_capture_safe] FAIL: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
