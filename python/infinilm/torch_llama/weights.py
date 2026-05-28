# Copyright (c) 2025, InfiniCore
"""Safetensors weight loading for torch_llama."""

from __future__ import annotations

import glob
import os
from typing import Dict

import torch
from safetensors import safe_open


def load_safetensors_state_dict(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Load all shards under ``model_path/*.safetensors`` into a flat state dict."""
    state: Dict[str, torch.Tensor] = {}
    for file_path in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).to(dtype=dtype)
    if not state:
        raise FileNotFoundError(f"No *.safetensors under {model_path}")
    if state.get("lm_head.weight") is None and "model.embed_tokens.weight" in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    for key, tensor in list(state.items()):
        state[key] = tensor.to(device=device)
    return state
