import math
import os
import ctypes
from typing import Callable, Tuple

import numpy as np

import infinicore
import infinicore as ic
from infinicore.nn import Module, Parameter
from infinicore.nn import functional as F

def _tensor_to_numpy(tensor: infinicore.Tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor

    try:
        cpu_dev = infinicore.device("cpu", 0)
        if str(tensor.device) != "cpu:0":
            cpu_tensor = tensor.to(cpu_dev)
        else:
            cpu_tensor = tensor

        dtype_str = str(cpu_tensor.dtype)
        if "float32" in dtype_str:
            CType = ctypes.c_float
            np_dtype = np.float32
        elif "bfloat16" in dtype_str or "half" in dtype_str or "float16" in dtype_str:
            CType = ctypes.c_uint16
            np_dtype = np.uint16
        elif "int64" in dtype_str:
            CType = ctypes.c_longlong
            np_dtype = np.int64
        elif "int32" in dtype_str:
            CType = ctypes.c_int
            np_dtype = np.int32
        else:
            CType = ctypes.c_float
            np_dtype = np.float32

        ptr = cpu_tensor.data_ptr()
        size = cpu_tensor.numel()

        ArrayType = CType * size
        c_array = ArrayType.from_address(ptr)
        np_arr = np.ctypeslib.as_array(c_array).copy().reshape(cpu_tensor.shape)

        if np_dtype == np.uint16:
            np_arr = np_arr.astype(np.float32)

        return np_arr
    except Exception as e:
        raise TypeError(
            f"Could not convert {type(tensor)} to numpy array. Direct access failed: {e}"
        )


def _from_numpy(
    array: np.ndarray,
    *,
    dtype: infinicore.dtype | None = None,
    device: infinicore.device | None = None,
) -> infinicore.Tensor:
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)

    # HACK: Direct Memory Injection (Write Hack)
    # infinicore.from_numpy() is broken for some versions/dtypes.
    # We create a container tensor and write data directly to its memory.
    
    try:
        # 1. Prepare source data (ensure float32 for safety)
        src_data = array.astype(np.float32)
        

        tensor = infinicore.zeros(
            list(src_data.shape),
            dtype=infinicore.float32,
            device=infinicore.device("cpu", 0),
        )
        
        # 3. Get pointer and inject data
        ptr = tensor.data_ptr()
        size = tensor.numel()
        
        DstType = ctypes.c_float * size
        dst_array = DstType.from_address(ptr)
        
        # Flatten and copy
        src_flat = src_data.flatten()
        # Using ctypes memmove might be faster, but loop is explicit
        # dst_array[:] = src_flat # This doesn't work directly on ctypes array
        
        # Copy data (using memmove for speed)
        ctypes.memmove(dst_array, src_flat.ctypes.data, src_flat.nbytes)

        # 4. Move to target device
        if device is not None and str(device) != "cpu:0":
            tensor = tensor.to(device)

        # 5. Cast to target dtype
        if dtype is not None and tensor.dtype != dtype:
             tensor = tensor.to(dtype=dtype)

        return tensor
        
    except Exception as e:
        raise TypeError(f"Could not create infinicore.Tensor from numpy array via data_ptr hack: {e}")


def _from_numpy_like(array: np.ndarray, like: infinicore.Tensor) -> infinicore.Tensor:
    return _from_numpy(array, dtype=like.dtype, device=like.device)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x - x_max)
    denom = np.sum(x_exp, axis=-1, keepdims=True)
    return x_exp / np.maximum(denom, 1e-9)


def _topk_np(prob: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("top-k must be positive.")
    # argpartition gives us the largest k elements in arbitrary order
    partition_idx = np.argpartition(prob, -k, axis=-1)[..., -k:]
    topk_values = np.take_along_axis(prob, partition_idx, axis=-1)
    # sort descending within the top-k slice
    sort_order = np.argsort(-topk_values, axis=-1)
    topk_indices = np.take_along_axis(partition_idx, sort_order, axis=-1)
    topk_values = np.take_along_axis(topk_values, sort_order, axis=-1)
    return topk_values, topk_indices


def _one_hot_np(indices: np.ndarray, num_classes: int) -> np.ndarray:
    eye = np.eye(num_classes, dtype=np.int64)
    return eye[indices]


def _activation_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    lower = name.lower()
    if lower in ("silu", "swish"):
        return lambda x: x / (1.0 + np.exp(-x))
    if lower == "gelu":
        return lambda x: 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))
    if lower == "relu":
        return lambda x: np.maximum(x, 0.0)
    raise KeyError(f"Unsupported activation '{name}' for Qwen3 MoE block.")


class Qwen3MoeExperts(Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(
        self,
        config,
        *,
        device: infinicore.device | None = None,
        dtype: infinicore.dtype | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        factory_kwargs = {
            "device": device if device is not None else infinicore.device("cpu", 0),
            "dtype": dtype if dtype is not None else infinicore.float32,
        }
        self.gate_up_proj = Parameter(
            infinicore.empty(
                [self.num_experts, 2 * self.intermediate_dim, self.hidden_dim],
                **factory_kwargs,
            )
        )
        self.down_proj = Parameter(
            infinicore.empty(
                [self.num_experts, self.hidden_dim, self.intermediate_dim],
                **factory_kwargs,
            )
        )
        self.act_fn = _activation_fn(getattr(config, "hidden_act", "silu"))

    def forward(
        self,
        hidden_states: infinicore.Tensor,
        top_k_index: infinicore.Tensor,
        top_k_weights: infinicore.Tensor,
    ) -> infinicore.Tensor:
        hidden_np = _tensor_to_numpy(hidden_states)
        topk_idx_np = _tensor_to_numpy(top_k_index).astype(np.int64, copy=False)
        topk_w_np = _tensor_to_numpy(top_k_weights)

        final_hidden_np = np.zeros_like(hidden_np)
        expert_mask = _one_hot_np(topk_idx_np, self.num_experts)  # [tokens, top_k, num_experts]
        expert_mask = np.transpose(expert_mask, (2, 1, 0))  # [num_experts, top_k, tokens]
        expert_hit = np.nonzero(np.sum(expert_mask, axis=(1, 2)) > 0)[0]

        gate_up_np = _tensor_to_numpy(self.gate_up_proj)
        down_proj_np = _tensor_to_numpy(self.down_proj)

        for expert_idx in expert_hit:
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = np.nonzero(expert_mask[expert_idx])
            if token_idx.size == 0:
                continue

            current_state = hidden_np[token_idx]
            # Gate + Up projection
            proj_out = current_state @ gate_up_np[expert_idx].T
            gate_part = proj_out[:, : self.intermediate_dim]
            up_part = proj_out[:, self.intermediate_dim :]

            activated = self.act_fn(gate_part) * up_part

            # Down projection
            current_hidden = activated @ down_proj_np[expert_idx].T
            current_hidden = current_hidden * topk_w_np[token_idx, top_k_pos][:, None]

            # Scatter-add back to the output buffer
            np.add.at(final_hidden_np, token_idx, current_hidden.astype(final_hidden_np.dtype, copy=False))

        return _from_numpy_like(final_hidden_np, hidden_states)


class Qwen3MoeTopKRouter(Module):
    def __init__(
        self,
        config,
        *,
        device: infinicore.device | None = None,
        dtype: infinicore.dtype | None = None,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        self.hidden_dim = config.hidden_size
        factory_kwargs = {
            "device": device if device is not None else infinicore.device("cpu", 0),
            "dtype": dtype if dtype is not None else infinicore.float32,
        }
        self.weight = Parameter(infinicore.zeros([self.num_experts, self.hidden_dim], **factory_kwargs))

    def forward(self, hidden_states: infinicore.Tensor):
        # Avoid negative-dim inference; some backends reject view with -1.
        tokens = int(hidden_states.shape[0])
        hidden_states = hidden_states.view((tokens, self.hidden_dim))
        router_logits = F.linear(hidden_states, self.weight)

        softmax_impl = os.environ.get("INFINILM_QWEN3MOE_ROUTER_SOFTMAX", "auto").lower()
        if softmax_impl not in ("auto", "ic", "numpy"):
            softmax_impl = "auto"

        router_prob = None
        router_prob_np = None
        can_use_ic = (
            infinicore.use_ntops
            and router_logits.device.type in ("cuda", "musa")
            and softmax_impl in ("auto", "ic")
        )

###########################################
        softmax_debug = os.environ.get("INFINILM_QWEN3MOE_ROUTER_SOFTMAX_DEBUG", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if softmax_debug and not getattr(self, "_softmax_debug_printed", False):
            selected = "ic" if can_use_ic else "numpy"
            print(
                "[Qwen3MoeTopKRouter] "
                f"softmax_impl={softmax_impl} selected={selected} "
                f"use_ntops={infinicore.use_ntops} device={router_logits.device}"
            )
            setattr(self, "_softmax_debug_printed", True)
###########################################

        if can_use_ic:
            router_prob = ic.softmax(router_logits, dim=-1)
            router_prob_np = _tensor_to_numpy(router_prob)
        else:
            if softmax_impl == "ic":
                raise RuntimeError(
                    "ic.softmax is only available with ntops on CUDA/MUSA devices; "
                    f"use_ntops={infinicore.use_ntops} device={router_logits.device}"
                )
            router_prob_np = _softmax_np(_tensor_to_numpy(router_logits))
            router_prob = _from_numpy(router_prob_np, dtype=router_logits.dtype, device=router_logits.device)

        router_top_value, router_indices = _topk_np(router_prob_np, self.top_k)
        if self.norm_topk_prob:
            denom = np.sum(router_top_value, axis=-1, keepdims=True)
            router_top_value = np.divide(router_top_value, denom, where=denom != 0)

        router_scores = _from_numpy(
            router_top_value.astype(router_prob_np.dtype, copy=False),
            dtype=router_prob.dtype,
            device=router_prob.device,
        )
        router_indices_tensor = _from_numpy(router_indices.astype(np.int64), device=router_prob.device)
        return router_prob, router_scores, router_indices_tensor


class Qwen3MoeSparseMoeBlock(Module):
    def __init__(
        self,
        config,
        *,
        device: infinicore.device | None = None,
        dtype: infinicore.dtype | None = None,
    ):
        super().__init__()
        self.experts = Qwen3MoeExperts(config, device=device, dtype=dtype)
        self.gate = Qwen3MoeTopKRouter(config, device=device, dtype=dtype)

    def forward(self, hidden_states: infinicore.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        tokens = int(batch_size * sequence_length)
        hidden_states_reshaped = hidden_states.view((tokens, hidden_dim))
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        top_k = int(routing_weights.shape[1])
        return (
            final_hidden_states.view((batch_size, sequence_length, hidden_dim)),
            routing_weights.view((batch_size, sequence_length, top_k)),
        )