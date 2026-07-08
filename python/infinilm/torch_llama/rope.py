# Copyright (c) 2025, InfiniCore
"""RoPE helpers that avoid dynamic control flow under torch.compile."""

from __future__ import annotations

import math
from typing import Optional

import torch


def _rope_scaling_dict(rotary_module: torch.nn.Module) -> Optional[dict]:
    scaling = getattr(rotary_module, "rope_scaling", None)
    if scaling is None:
        config = getattr(rotary_module, "config", None)
        scaling = getattr(config, "rope_scaling", None) if config is not None else None
    return scaling


def _is_longrope(scaling: Optional[dict]) -> bool:
    if not scaling:
        return False
    rope_type = scaling.get("rope_type") or scaling.get("type")
    return rope_type == "longrope"


def _longrope_table_factor(scaling: dict) -> float:
    """Match InfiniCore ``LongRopeConfig::factor()`` (``rope.cc``)."""
    raw = float(scaling.get("factor", 1.0))
    if raw == 1.0:
        return 1.0
    boundary = int(scaling["original_max_position_embeddings"])
    return math.sqrt(1.0 + math.log(raw) / math.log(float(boundary)))


def _longrope_inv_freq(
    rotary_module: torch.nn.Module,
    position_ids: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
  Position-dependent inverse frequencies for LongRoPE.

  Returns ``[batch, seq_len, head_dim/2]`` matching InfiniCore cache rows.
  """
    scaling = _rope_scaling_dict(rotary_module)
    if scaling is None:
        raise ValueError("longrope scaling config missing")

    short_factor = torch.tensor(
        scaling["short_factor"], device=device, dtype=dtype
    )
    long_factor = torch.tensor(
        scaling["long_factor"], device=device, dtype=dtype
    )
    cache_dim = int(short_factor.numel())
    head_dim = cache_dim * 2
    theta = float(getattr(rotary_module.config, "rope_theta", 10000.0))
    boundary = float(scaling["original_max_position_embeddings"])

    j = torch.arange(cache_dim, device=device, dtype=dtype)
    base = torch.pow(
        torch.tensor(theta, device=device, dtype=dtype),
        2.0 * j / float(head_dim),
    )

    pos = position_ids.to(device=device, dtype=dtype)
    use_short = pos < boundary
    ext = torch.where(
        use_short.unsqueeze(-1),
        short_factor.view(1, 1, -1),
        long_factor.view(1, 1, -1),
    )
    return 1.0 / (ext * base.view(1, 1, -1))


def rotary_embeddings_compile_friendly(
    rotary_module: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin without ``dynamic_rope_update`` (LongRoPE branch on ``seq_len``).

    Supports LongRoPE ``short_factor`` / ``long_factor`` / ``factor`` semantics
    aligned with InfiniCore ``rope.cc`` (lines 94–114).
    """
    scaling = _rope_scaling_dict(rotary_module)
    device_type = (
        hidden.device.type
        if isinstance(hidden.device.type, str) and hidden.device.type != "mps"
        else "cpu"
    )

    with torch.autocast(device_type=device_type, enabled=False):
        if _is_longrope(scaling):
            inv_freq_pos = _longrope_inv_freq(
                rotary_module,
                position_ids,
                device=hidden.device,
                dtype=torch.float32,
            )
            freqs = inv_freq_pos * position_ids.to(
                device=hidden.device, dtype=torch.float32
            ).unsqueeze(-1)
            table_factor = _longrope_table_factor(scaling)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * table_factor
            sin = emb.sin() * table_factor
        else:
            inv_freq = getattr(rotary_module, "_inv_freq_f32", None)
            if inv_freq is None:
                inv_freq_src = rotary_module.original_inv_freq
                if inv_freq_src.device.type == "meta":
                    inv_freq_src = rotary_module.inv_freq
                inv_freq = inv_freq_src
                if inv_freq.dtype != torch.float32:
                    inv_freq = inv_freq.float()
                if inv_freq.device != hidden.device:
                    inv_freq = inv_freq.to(hidden.device)
            inv_freq_expanded = inv_freq[None, :, None].expand(
                position_ids.shape[0], -1, 1
            )
            position_ids_expanded = position_ids[:, None, :].float()
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * rotary_module.attention_scaling
            sin = emb.sin() * rotary_module.attention_scaling

    return cos.to(dtype=hidden.dtype), sin.to(dtype=hidden.dtype)


def segment_position_embeddings(
    rotary_module: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    valid_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin for piecewise pre-attn staging RoPE (valid prefix only)."""
    valid_len = int(valid_len)
    if valid_len <= 0:
        raise ValueError("valid_len must be positive for segment RoPE")
    pos = position_ids[:, :valid_len]
    return rotary_embeddings_compile_friendly(rotary_module, hidden, pos)
