# Copyright (c) 2025, InfiniCore
"""Custom op: opaque FusedMoE boundary for MoE AOT segments (serve vLLM-free)."""

from __future__ import annotations

from typing import Optional

import torch

_LIB: Optional[torch.library.Library] = None
_REGISTERED = False

FUSED_MOE_ROUTED_OP = "infinilm.fused_moe_routed"


def _run_fused_moe_routed(
    x: torch.Tensor,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    from infinilm.kernels.fused_moe_runtime import fused_moe_routed

    return fused_moe_routed(x, topk_w, topk_ids, w_gate_up, w_down)


def register_fused_moe_routed_op() -> None:
    """Idempotent registration of ``infinilm.fused_moe_routed``."""
    global _LIB, _REGISTERED
    if _REGISTERED:
        return

    _LIB = torch.library.Library("infinilm", "FRAGMENT")
    _LIB.define(
        "fused_moe_routed("
        "Tensor x, Tensor topk_w, Tensor topk_ids, "
        "Tensor w_gate_up, Tensor w_down"
        ") -> Tensor"
    )

    @torch.library.impl("infinilm::fused_moe_routed", "CUDA")
    def _fused_moe_cuda(
        x: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        return _run_fused_moe_routed(x, topk_w, topk_ids, w_gate_up, w_down)

    @torch.library.impl("infinilm::fused_moe_routed", "PrivateUse1")
    def _fused_moe_maca(
        x: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        return _run_fused_moe_routed(x, topk_w, topk_ids, w_gate_up, w_down)

    @torch.library.register_fake("infinilm::fused_moe_routed")
    def _fused_moe_fake(
        x: torch.Tensor,
        topk_w: torch.Tensor,
        topk_ids: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        del topk_w, topk_ids, w_gate_up, w_down
        return torch.empty_like(x)

    _REGISTERED = True
