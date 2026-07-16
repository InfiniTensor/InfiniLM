#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""G1: MoE custom-op export / AOT smoke (fused_moe_routed opaque boundary)."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    from infinilm.tools.gate_common import (
        cache_root_default,
        model_path_default,
        write_gate_result,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", type=int, default=512)
    ap.add_argument(
        "--cache-root",
        default=str(cache_root_default()),
    )
    ap.add_argument("--model-path", default=model_path_default())
    ap.add_argument(
        "--skip-aot",
        action="store_true",
        help="Only export+inspect graph (no AOT package write)",
    )
    args = ap.parse_args()

    details: dict = {"bucket": args.bucket, "model_path": args.model_path}
    try:
        import torch
        from infinilm.torch_llama.moe_ops import (
            FUSED_MOE_ROUTED_OP,
            register_fused_moe_routed_op,
        )

        register_fused_moe_routed_op()
        details["op_registered"] = True

        # Minimal module that only exercises the opaque op (export smoke).
        class _RoutedOnly(torch.nn.Module):
            def forward(self, x, topk_w, topk_ids, w_gate_up, w_down):
                return torch.ops.infinilm.fused_moe_routed(
                    x, topk_w, topk_ids, w_gate_up, w_down
                )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        M, H, E, N, K = int(args.bucket), 2048, 160, 512, 8
        if device.type != "cuda":
            # Fake tensors for CPU export structure check
            M = min(M, 16)
        x = torch.randn(M, H, device=device, dtype=dtype)
        topk_w = torch.randn(M, K, device=device, dtype=dtype)
        topk_ids = torch.zeros(M, K, device=device, dtype=torch.int32)
        w_gu = torch.randn(E, 2 * N, H, device=device, dtype=dtype)
        w_d = torch.randn(E, H, N, device=device, dtype=dtype)
        mod = _RoutedOnly().eval()
        exported = torch.export.export(
            mod, (x, topk_w, topk_ids, w_gu, w_d), strict=False
        )
        graph_str = str(exported.graph)
        details["graph_has_fused_moe_routed"] = (
            "fused_moe_routed" in graph_str or FUSED_MOE_ROUTED_OP in graph_str
        )
        details["graph_has_index_select"] = "index_select" in graph_str
        if not details["graph_has_fused_moe_routed"]:
            raise RuntimeError(
                "exported graph missing infinilm.fused_moe_routed; "
                f"graph snippet={graph_str[:800]}"
            )
        # Opaque op body must not expand to index_select routed experts.
        if details["graph_has_index_select"]:
            raise RuntimeError(
                "exported graph still contains index_select (legacy routed body)"
            )

        if not args.skip_aot and device.type == "cuda":
            from infinilm.compile.piecewise_moe_segment import (
                aot_compile_minicpm5_moe_segment,
            )

            os.environ.setdefault("INFINI_PIECEWISE_LAYER_AGNOSTIC", "1")
            meta = aot_compile_minicpm5_moe_segment(
                model_path=args.model_path,
                bucket=int(args.bucket),
                device=device,
                cache_root=args.cache_root,
                require_aot=True,
            )
            details["aot"] = {
                "package_path": meta.get("package_path"),
                "artifact_dir": meta.get("artifact_dir"),
            }
            pkg = meta.get("package_path")
            if not pkg or not os.path.isfile(pkg):
                raise RuntimeError(f"AOT package missing after compile: {pkg}")

        path = write_gate_result("G1", status="PASS", details=details)
        print(f"[G1] PASS → {path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        path = write_gate_result("G1", status="FAIL", details=details, error=str(exc))
        print(f"[G1] FAIL → {path}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
