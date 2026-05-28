# Copyright (c) 2025, InfiniCore
"""Minimal 9g/Llama prefill module using transformers + flash-attn (eager path)."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class TorchLlamaPrefillModel(nn.Module):
    """Eager PyTorch decoder for prefill; ``forward`` returns full-sequence logits."""

    def __init__(self, inner: AutoModelForCausalLM):
        super().__init__()
        self.inner = inner

    @property
    def config(self):
        return self.inner.config

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: ``[batch, seq_len]`` int64 on device.
        Returns:
            Logits ``[batch, seq_len, vocab]`` in model dtype.
        """
        out = self.inner(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )
        return out.logits

    def forward_prefill_compile(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prefill forward without ``DynamicCache`` / mask helpers (torch.compile friendly)."""
        from .rope import rotary_embeddings_compile_friendly

        model = self.inner.model
        hidden = model.embed_tokens(input_ids)
        seq_len = hidden.shape[1]
        device = hidden.device
        cache_position = torch.arange(0, seq_len, device=device, dtype=torch.long)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = rotary_embeddings_compile_friendly(
            model.rotary_emb, hidden, position_ids
        )

        for decoder_layer in model.layers:
            hidden = decoder_layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden = model.norm(hidden)
        return self.inner.lm_head(hidden)

    def forward_last_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits at the last prefill position: ``[batch, vocab]``."""
        logits = self.forward(input_ids)
        return logits[:, -1, :]


def _materialize_meta_rotary_buffers(inner: AutoModelForCausalLM) -> None:
    """Copy ``inv_freq`` into ``original_inv_freq`` when meta-init left the latter empty."""
    rotary = getattr(getattr(inner, "model", inner), "rotary_emb", None)
    if rotary is None or not hasattr(rotary, "original_inv_freq"):
        return
    original = rotary.original_inv_freq
    if original.device.type != "meta":
        return
    device = rotary.inv_freq.device
    rotary.original_inv_freq = rotary.inv_freq.to(device=device).clone()


def load_torch_llama(
    model_path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "flash_attention_2",
    splitting_flash_boundary: bool = False,
    cpp_state_dict: Optional[dict] = None,
) -> TorchLlamaPrefillModel:
    """Build model, load HF weights from safetensors, eval mode on ``device``.

    When ``cpp_state_dict`` is provided, allocate an empty module on ``device``
    and bind zero-copy views into C++ buffers (hybrid InferEngine path).
    """
    model_path = os.path.expanduser(model_path)
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if dtype is None:
        torch_dtype_name = config_dict.get("torch_dtype") or config_dict.get(
            "dtype", "bfloat16"
        )
        dtype = getattr(torch, torch_dtype_name, torch.bfloat16)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cpp_state_dict is not None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        with torch.device("meta"):
            inner = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        inner = inner.to_empty(device=device)
        inner.eval()
        if splitting_flash_boundary:
            from .attention import enable_splitting_flash_on_model

            enable_splitting_flash_on_model(inner)
        wrapper = TorchLlamaPrefillModel(inner)
        from infinilm.compile.weights import bind_cpp_weights_to_torch

        bind_cpp_weights_to_torch(wrapper, cpp_state_dict, strict=False)
        _materialize_meta_rotary_buffers(inner)
        return wrapper

    inner = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    inner = inner.to(device=device, dtype=dtype)
    inner.eval()
    if splitting_flash_boundary:
        from .attention import enable_splitting_flash_on_model

        enable_splitting_flash_on_model(inner)
    return TorchLlamaPrefillModel(inner)
