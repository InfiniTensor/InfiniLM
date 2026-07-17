# Copyright (c) 2025, InfiniCore
"""Custom ops: opaque FusedMoE + decode-eager MoE block (serve vLLM-free)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

_LIB: Optional[torch.library.Library] = None
_REGISTERED = False

FUSED_MOE_ROUTED_OP = "infinilm.fused_moe_routed"
MOE_BLOCK_EAGER_OP = "infinilm.moe_block_eager"

# MiniCPM5 defaults (config.json); bootstrap overrides from HF.
_ROUTING: Dict[str, Any] = {
    "top_k": 16,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 3.66,
}


def configure_moe_block_routing(
    *,
    top_k: int,
    n_group: int = 1,
    topk_group: int = 1,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Set routing hparams for ``moe_block_eager`` (decode AOTI bypass)."""
    _ROUTING["top_k"] = int(top_k)
    _ROUTING["n_group"] = int(n_group)
    _ROUTING["topk_group"] = int(topk_group)
    _ROUTING["norm_topk_prob"] = bool(norm_topk_prob)
    _ROUTING["routed_scaling_factor"] = float(routed_scaling_factor)


def moe_block_routing() -> Dict[str, Any]:
    return dict(_ROUTING)


def _run_fused_moe_routed(
    x: torch.Tensor,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    from infinilm.kernels.fused_moe_runtime import fused_moe_routed

    return fused_moe_routed(x, topk_w, topk_ids, w_gate_up, w_down)


def _silu_mlp(x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    gu = F.linear(x, w_gate_up)
    gate, up = gu.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, w_down)


def moe_block_eager(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    shared_gate_up: torch.Tensor,
    shared_down: torch.Tensor,
) -> torch.Tensor:
    """Full MoE block for decode (no AOTI): router + Triton routed + shared MLP.

    x: [T, H] valid tokens (already unpadded). Returns [T, H].
    """
    from infinilm.compile.piecewise_moe_segment import grouped_sigmoid_topk

    cfg = _ROUTING
    logits = F.linear(
        x.to(dtype=torch.float32), gate_weight.to(dtype=torch.float32)
    )
    topk_w, topk_ids = grouped_sigmoid_topk(
        logits,
        e_score_correction_bias,
        top_k=int(cfg["top_k"]),
        n_group=int(cfg["n_group"]),
        topk_group=int(cfg["topk_group"]),
        norm_topk_prob=bool(cfg["norm_topk_prob"]),
        routed_scaling_factor=float(cfg["routed_scaling_factor"]),
    )
    # Match warmed Triton cubins (bf16/fp16 weights, int32 ids).
    topk_w = topk_w.to(dtype=x.dtype)
    topk_ids = topk_ids.to(dtype=torch.int32)
    routed = _run_fused_moe_routed(x, topk_w, topk_ids, w_gate_up, w_down)
    shared = _silu_mlp(x, shared_gate_up, shared_down)
    return routed + shared


def register_fused_moe_routed_op() -> None:
    """Idempotent registration of ``fused_moe_routed`` + ``moe_block_eager``."""
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
    _LIB.define(
        "moe_block_eager("
        "Tensor x, Tensor gate_weight, Tensor e_score_correction_bias, "
        "Tensor w_gate_up, Tensor w_down, "
        "Tensor shared_gate_up, Tensor shared_down"
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

    @torch.library.impl("infinilm::moe_block_eager", "CUDA")
    def _moe_block_cuda(
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        shared_gate_up: torch.Tensor,
        shared_down: torch.Tensor,
    ) -> torch.Tensor:
        return moe_block_eager(
            x,
            gate_weight,
            e_score_correction_bias,
            w_gate_up,
            w_down,
            shared_gate_up,
            shared_down,
        )

    @torch.library.impl("infinilm::moe_block_eager", "PrivateUse1")
    def _moe_block_maca(
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        shared_gate_up: torch.Tensor,
        shared_down: torch.Tensor,
    ) -> torch.Tensor:
        return moe_block_eager(
            x,
            gate_weight,
            e_score_correction_bias,
            w_gate_up,
            w_down,
            shared_gate_up,
            shared_down,
        )

    @torch.library.register_fake("infinilm::moe_block_eager")
    def _moe_block_fake(
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        shared_gate_up: torch.Tensor,
        shared_down: torch.Tensor,
    ) -> torch.Tensor:
        del (
            gate_weight,
            e_score_correction_bias,
            w_gate_up,
            w_down,
            shared_gate_up,
            shared_down,
        )
        return torch.empty_like(x)

    _REGISTERED = True
