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

    def forward_prefill_compile(
        self,
        input_ids: torch.Tensor,
        *,
        valid_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Prefill forward without ``DynamicCache`` / mask helpers (torch.compile friendly)."""
        from .prefill_context import prefill_compile_context
        from .rope import rotary_embeddings_compile_friendly

        model = self.inner.model
        bucket_len = int(input_ids.shape[1])
        actual_len = int(valid_seq_len) if valid_seq_len is not None else bucket_len

        with prefill_compile_context(actual_len):
            hidden = model.embed_tokens(input_ids)
            device = hidden.device
            cache_position = torch.arange(0, bucket_len, device=device, dtype=torch.long)
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


def _materialize_rotary_for_compile(inner: AutoModelForCausalLM) -> None:
    """Float32 ``original_inv_freq`` on device for capture-safe RoPE (no Inductor copy)."""
    rotary = getattr(getattr(inner, "model", inner), "rotary_emb", None)
    if rotary is None:
        return
    device = rotary.inv_freq.device
    if device.type == "meta":
        device = next(inner.parameters()).device
    inv_f32 = rotary.inv_freq.to(device=device, dtype=torch.float32).contiguous()
    if hasattr(rotary, "original_inv_freq"):
        rotary.original_inv_freq = inv_f32.clone()
    rotary.register_buffer("_inv_freq_f32", inv_f32, persistent=False)


def load_torch_llama(
    model_path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "flash_attention_2",
    splitting_flash_boundary: bool = False,
    cpp_state_dict: Optional[dict] = None,
    tp_size: int = 1,
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
        if int(tp_size) > 1:
            tp_size = int(tp_size)
            config.num_attention_heads = int(config.num_attention_heads) // tp_size
            total_kv = int(config.num_key_value_heads)
            config.num_key_value_heads = (
                1 if total_kv < tp_size else total_kv // tp_size
            )
            config.intermediate_size = int(config.intermediate_size) // tp_size
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

        bind_cpp_weights_to_torch(wrapper, cpp_state_dict, strict=False, device=device)
        _materialize_rotary_for_compile(inner)
        from .mup import Fm9gMupScales, apply_fm9g_mup_runtime_scales

        mup = Fm9gMupScales.from_config(config)
        if mup is not None:
            apply_fm9g_mup_runtime_scales(inner, mup)
            wrapper._fm9g_mup_scales = mup
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
    _materialize_rotary_for_compile(inner)
    return TorchLlamaPrefillModel(inner)
