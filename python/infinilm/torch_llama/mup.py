# Copyright (c) 2025, InfiniCore
"""FM9G MuP runtime alphas matching C++ ``BaseLinear::alpha_`` (PRD-04 M3)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn

_FM9G_MODEL_TYPES = frozenset({"fm9g", "fm9g7b", "minicpm"})


@dataclass(frozen=True)
class Fm9gMupScales:
    """Runtime multipliers applied when C++ shares unscaled weight buffers."""

    lm_head_alpha: float
    proj_alpha: float

    @classmethod
    def from_config(cls, config: Any) -> Optional[Fm9gMupScales]:
        """Return MuP scales when config or env indicates FM9G-style runtime alphas."""
        if isinstance(config, dict):
            model_type = config.get("model_type")
            scale_depth_raw = config.get("scale_depth")
            num_hidden_layers = int(config.get("num_hidden_layers", 1))
            hidden_size = int(config.get("hidden_size", 1))
            dim_model_base = config.get("dim_model_base")
        else:
            model_type = getattr(config, "model_type", None)
            scale_depth_raw = getattr(config, "scale_depth", None)
            num_hidden_layers = int(config.num_hidden_layers)
            hidden_size = int(config.hidden_size)
            dim_model_base = getattr(config, "dim_model_base", None)

        scale_depth = 1.0 if scale_depth_raw is None else float(scale_depth_raw)
        has_mup_fields = scale_depth != 1.0 or dim_model_base is not None
        env_force = os.environ.get("INFINI_FM9G_MUP", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if (
            model_type not in _FM9G_MODEL_TYPES
            and not has_mup_fields
            and not env_force
        ):
            return None

        proj_alpha = 1.0
        if scale_depth != 1.0:
            proj_alpha = scale_depth / math.sqrt(float(num_hidden_layers))

        lm_head_alpha = 1.0
        if dim_model_base is not None:
            lm_head_alpha = float(dim_model_base) / float(hidden_size)

        if proj_alpha == 1.0 and lm_head_alpha == 1.0 and not env_force:
            return None

        return cls(lm_head_alpha=lm_head_alpha, proj_alpha=proj_alpha)


def _wrap_module_forward(module: nn.Module, alpha: float) -> None:
    if alpha == 1.0:
        return
    original = module.forward

    def forward(*args, **kwargs):
        return alpha * original(*args, **kwargs)

    module.forward = forward  # type: ignore[method-assign]


def apply_fm9g_mup_runtime_scales(inner: nn.Module, scales: Fm9gMupScales) -> None:
    """Match ``FM9GForCausalLM`` / ``FM9GAttention`` / ``FM9GMLP`` C++ ``set_alpha``."""
    layers = getattr(getattr(inner, "model", inner), "layers", None)
    if layers is None:
        raise ValueError("expected HuggingFace LlamaForCausalLM-style module")

    for layer in layers:
        _wrap_module_forward(layer.self_attn.o_proj, scales.proj_alpha)
        _wrap_module_forward(layer.mlp.down_proj, scales.proj_alpha)

    _wrap_module_forward(inner.lm_head, scales.lm_head_alpha)
