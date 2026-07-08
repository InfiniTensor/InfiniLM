# Copyright (c) 2025, InfiniCore
"""Share C++ InferEngine weight buffers with the torch prefill backbone (PRD-04 M3)."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import infinicore
import torch
from infinicore.utils import to_torch_dtype

from infinilm.torch_llama.model import TorchLlamaPrefillModel

logger = logging.getLogger(__name__)


def _wrap_infini_tensor(infini_tensor) -> infinicore.Tensor:
    if hasattr(infini_tensor, "_underlying"):
        return infini_tensor
    return infinicore.Tensor(infini_tensor)


def _try_zero_copy_view(
    infini_tensor,
    *,
    shape,
    dtype,
    device: torch.device,
) -> torch.Tensor | None:
    if not hasattr(torch, "from_blob"):
        return None
    wrapped = _wrap_infini_tensor(infini_tensor)
    u = wrapped._underlying
    if device.type == "cuda":
        blob_device = torch.device("cuda", device.index if device.index is not None else 0)
    else:
        blob_device = device
    return torch.from_blob(u.data_ptr(), shape, dtype=dtype, device=blob_device)


def bind_cpp_weights_to_torch(
    torch_model: TorchLlamaPrefillModel,
    cpp_state_dict: Dict[str, object],
    *,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> int:
    """
    Point torch parameters at C++ buffers (zero-copy when supported, else one-time copy).

    Returns the number of parameters rebound.
    """
    if device is None:
        device = next(torch_model.inner.parameters()).device

    rebound = 0
    missing = []
    for name, param in torch_model.inner.named_parameters():
        if name not in cpp_state_dict:
            missing.append(name)
            continue
        src = _wrap_infini_tensor(cpp_state_dict[name])
        dtype = to_torch_dtype(src.dtype)
        shape = tuple(src.shape)
        view = _try_zero_copy_view(src, shape=shape, dtype=dtype, device=device)
        if view is not None:
            if tuple(view.shape) != tuple(param.shape):
                if strict:
                    raise RuntimeError(
                        f"weight shape mismatch for {name}: cpp={tuple(view.shape)} "
                        f"torch={tuple(param.shape)}"
                    )
                missing.append(name)
                continue
            param.data = view
        else:
            if tuple(shape) != tuple(param.shape):
                if strict:
                    raise RuntimeError(
                        f"weight shape mismatch for {name}: cpp={shape} torch={tuple(param.shape)}"
                    )
                missing.append(name)
                continue
            infinicore.from_torch(param).copy_(src)
        rebound += 1

    if strict and missing:
        raise RuntimeError(f"cpp state_dict missing torch params: {missing[:8]}...")
    logger.info("shared %d torch prefill params from C++ buffers", rebound)
    return rebound
