# Copyright (c) 2025, InfiniCore
"""RoPE helpers that avoid dynamic control flow under torch.compile."""

from __future__ import annotations

import torch


def rotary_embeddings_compile_friendly(
    rotary_module: torch.nn.Module,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cos/sin without ``dynamic_rope_update`` (LongRoPE branch on ``seq_len``).

    Safe when prefill ``seq_len <= original_max_position_embeddings`` (9g: 65536;
    compile ladder tops out at 8192).
    """
    inv_freq_src = rotary_module.original_inv_freq
    if inv_freq_src.device.type == "meta":
        # ``to_empty`` materializes ``inv_freq`` but leaves ``original_inv_freq`` on meta
        # (shared-weight / meta-init path).
        inv_freq_src = rotary_module.inv_freq
    inv_freq = inv_freq_src.to(hidden.device)
    inv_freq_expanded = (
        inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    )
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = (
        hidden.device.type
        if isinstance(hidden.device.type, str) and hidden.device.type != "mps"
        else "cpu"
    )
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * rotary_module.attention_scaling
        sin = emb.sin() * rotary_module.attention_scaling

    return cos.to(dtype=hidden.dtype), sin.to(dtype=hidden.dtype)
