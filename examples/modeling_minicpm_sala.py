# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch MiniCPMSALA model."""
import os
import math
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from einops import rearrange, repeat
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS,
    is_torch_greater_or_equal_than_1_13,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
from fla.utils import tensor_cache


from .configuration_minicpm_sala import MiniCPMSALAConfig

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from infllm_v2 import (
        infllmv2_attn_stage1,
        infllmv2_attn_varlen_func,
        infllmv2_attn_with_kvcache,
        max_pooling_1d,
        max_pooling_1d_varlen,
    )
except ImportError:
    pass

from functools import lru_cache

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MiniCPMSALAConfig"


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    k2: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    cache_lens=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        batch_size = cu_seqlens_q.shape[0] - 1

        # Check if it's prefilling stage
        is_prefilling = cache_lens is None or (cache_lens == 0).all().item()

        if is_prefilling:  # prefilling stage
            # Calculate q_idx for each query position in each batch
            cache_lens = torch.zeros(batch_size, dtype=torch.int32, device=q.device)
            q_idx = torch.cat(
                [
                    (
                        torch.arange(
                            cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device
                        )
                        + max_seqlen_q
                        - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
                    )
                    // block_size
                    for i in range(batch_size)
                ],
                dim=0,
            )  # shape: [total_q_len]
        else:  # decoding stage
            # Each batch has only one query (last position)
            q_idx = (
                cache_lens // block_size
            )  # shape: [batch_size] = [total_q_len] in decoding

        # Compute attention score
        score = infllmv2_attn_stage1(
            q.contiguous(),
            k.contiguous(),
            k2.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_v=cu_seqlens_k2,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=is_prefilling,
        )
        score = score[:, : q_idx.shape[0], :]  # [num_heads, total_q_len, num_blocks]

        block_score = max_pooling_1d_varlen(
            score.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            cache_lens,
            max_seqlen_q,
            max_seqlen_k,
            local_blocks=local_blocks,
            init_blocks=init_blocks,
            block_size=block_size,
            stride=kernel_stride,
        )  # shape: [num_heads, total_q_len, num_blocks]

        # get topk
        topk = min(topk, block_score.shape[-1])
        topk_idx = block_score.topk(topk, dim=-1).indices.sort(-1).values
        topk_idx[topk_idx > q_idx[None, :, None]] = -1
        topk_idx = topk_idx.to(torch.int32)

    return topk_idx


@lru_cache(maxsize=16)
def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require Sparse attention, with stride support.

    Args:
        cu_seqlen (torch.Tensor): Cumulative sequence lengths for each sample.
        chunk_size (int): Chunk size used for Sparse attention.
        kernel_stride (int): Stride size when sliding over the sequence.

    Returns:
        filtered_indices (torch.Tensor): Indices used to directly index into the key/value tensors.
        cu_seqlens_compressed (torch.Tensor): Cumulative sequence lengths after compression.
    """
    # 1. Compute the length of each sequence
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(
        0,
        max_num_chunks_per_seq * kernel_stride,
        kernel_stride,
        device=cu_seqlen.device,
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = (
        seq_starts[:, None] + chunk_start_offsets[None, :]
    )  # [batch_size, max_num_chunks_per_seq]

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size
    valid_chunk_mask = chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks]
    del chunk_start_in_seq
    # 5. Generate filtered_indices
    chunk_indices = torch.arange(0, chunk_size, device=cu_seqlen.device)[
        None, :
    ]  # [1, chunk_size]
    filtered_indices = (
        valid_chunk_starts[:, None] + chunk_indices
    )  # [num_valid_chunks, chunk_size]
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(
        dim=1
    )  # Number of valid chunks per batch
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    del (
        num_filtered_chunks_per_batch,
        chunk_start_offsets,
        seq_starts,
        chunk_end_in_seq,
        valid_chunk_mask,
        chunk_indices,
    )
    return filtered_indices, cu_seqlens_compressed


class CompressK(torch.nn.Module):
    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        """
        Module for compressing key (K) representations.

        Args:
            head_num_k (int): Number of key attention heads.
            head_dim (int): Dimension of each attention head.
            kernel_size (int): Size of each chunk used for compression.
            kernel_stride (int, optional): Stride used when dividing input into chunks. Default is 16.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, k: torch.Tensor, cu_seqlens):
        """
        Forward pass for compressing the key (K) tensor.

        Args:
            k (torch.Tensor): Input key tensor of shape (total_seq_len, num_heads, head_dim).
            cu_seqlens (torch.Tensor): Cumulative sequence lengths for each sample in the batch, typically used for handling variable-length sequences.

        Returns:
            compress_k (torch.Tensor): Compressed key tensor.
            cu_seqlens_compressed (torch.Tensor): Updated cumulative sequence lengths after compression.

        """
        # Compute chunk-related metadata, with stride support
        filtered_k_indices, cu_seqlens_compressed = calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )

        # Extract filtered key vectors
        filtered_k = k.index_select(0, filtered_k_indices.view(-1))

        # split
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size,
            self.kernel_size,
            self.head_num_k,
            self.head_dim,
        )  # [l, block_size,h,d]

        compressed_k = filtered_k.mean(dim=1)
        return compressed_k, cu_seqlens_compressed


class InfLLMv2CacheLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        # Initialize any additional attributes specific to InfLLMv2CacheLayer
        self.no_rope_keys = torch.tensor([], dtype=torch.float32)
        self.compress_k_cache = []
        self.no_compress_k_cache = []
        self.cached_compressed_cu_seqlens = torch.tensor([], dtype=torch.int32)
        self.compress_k_cache_varlen = torch.tensor([], dtype=torch.float32)
        # Add support for compress_k2
        self.compress_k2_cache = []
        self.cached_compressed_cu_seqlens2 = torch.tensor([], dtype=torch.int32)
        self.compress_k2_cache_varlen = torch.tensor([], dtype=torch.float32)
        self.no_compress_k2_cache = []

    def update_no_rope_key(self, key_states):
        if self.no_rope_keys.numel() == 0:
            self.no_rope_keys = key_states
        else:
            self.no_rope_keys = torch.cat([self.no_rope_keys, key_states], dim=1)
        return self.no_rope_keys

    def update_compress_k(self, key_states, cu_seqlens=None):
        if len(self.compress_k_cache) == 0:
            if cu_seqlens is not None:
                self.cached_compressed_cu_seqlens = cu_seqlens.clone()
            self.compress_k_cache_varlen = key_states
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            self.compress_k_cache = list(torch.split(key_states, split_sizes))
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    self.compress_k_cache[index] = torch.cat(
                        [self.compress_k_cache[index], k], dim=0
                    )
            new_seq_lens = torch.tensor(
                [tensor.shape[0] for tensor in self.compress_k_cache], dtype=torch.int32
            )
            new_cumsum = torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32)

            self.compress_k_cache_varlen = torch.cat(self.compress_k_cache, dim=0)
            self.cached_compressed_cu_seqlens = torch.cat(
                [torch.tensor([0], dtype=torch.int32), new_cumsum]
            ).to(self.compress_k_cache_varlen.device)
        return self.compress_k_cache_varlen, self.cached_compressed_cu_seqlens

    def update_no_compress_k(self, key_states, kernel_size=32, kernel_stride=16):
        k_chunk_list = []
        for index, k in enumerate(key_states):
            if len(self.no_compress_k_cache) <= index:
                self.no_compress_k_cache.append(k)
            else:
                self.no_compress_k_cache[index] = torch.cat(
                    [self.no_compress_k_cache[index], k], dim=0
                )
                current_len = self.no_compress_k_cache[index].shape[0]
                if current_len >= kernel_size:
                    k_chunk_list.append(self.no_compress_k_cache[index][:kernel_size])
                    self.no_compress_k_cache[index] = self.no_compress_k_cache[index][
                        kernel_stride:
                    ]
                else:
                    k_chunk_list.append(None)
        return k_chunk_list

    def update_compress_k2(self, key_states, cu_seqlens=None):
        if len(self.compress_k2_cache) == 0:
            if cu_seqlens is not None:
                self.cached_compressed_cu_seqlens2 = cu_seqlens.clone()
            self.compress_k2_cache_varlen = key_states
            split_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            self.compress_k2_cache = list(torch.split(key_states, split_sizes))
        else:
            for index, k in enumerate(key_states):
                if k is not None:
                    self.compress_k2_cache[index] = torch.cat(
                        [self.compress_k2_cache[index], k], dim=0
                    )
            new_seq_lens = torch.tensor(
                [tensor.shape[0] for tensor in self.compress_k2_cache],
                dtype=torch.int32,
            )
            new_cumsum = torch.cumsum(new_seq_lens, dim=0, dtype=torch.int32)

            self.compress_k2_cache_varlen = torch.cat(self.compress_k2_cache, dim=0)
            self.cached_compressed_cu_seqlens2 = torch.cat(
                [torch.tensor([0], dtype=torch.int32), new_cumsum]
            ).to(self.compress_k2_cache_varlen.device)
        return self.compress_k2_cache_varlen, self.cached_compressed_cu_seqlens2

    def update_no_compress_k2(self, key_states, kernel_size=128, kernel_stride=64):
        k_chunk_list = []
        for index, k in enumerate(key_states):
            if len(self.no_compress_k2_cache) <= index:
                self.no_compress_k2_cache.append(k)
            else:
                self.no_compress_k2_cache[index] = torch.cat(
                    [self.no_compress_k2_cache[index], k], dim=0
                )
                current_len = self.no_compress_k2_cache[index].shape[0]
                if current_len >= kernel_size:
                    k_chunk_list.append(self.no_compress_k2_cache[index][:kernel_size])
                    self.no_compress_k2_cache[index] = self.no_compress_k2_cache[index][
                        kernel_stride:
                    ]
                else:
                    k_chunk_list.append(None)
        return k_chunk_list


class LightningCacheLayer(DynamicLayer):
    def __init__(self):
        super().__init__()
        self.state = {}

    def update(
        self,
        recurrent_state: torch.Tensor = None,
        attn_state: Tuple[torch.Tensor, torch.Tensor] = None,
        conv_state: Tuple[torch.Tensor] = None,
        ffn_state: torch.Tensor = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            recurrent_state (`torch.Tensor`, `optional`):
                The new recurrent state to cache.
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            conv_state (`Tuple[torch.Tensor]`, `optional`):
                The new convolution state to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        # Update the number of seen tokens

        if recurrent_state is not None:
            self.state["recurrent_state"] = recurrent_state
        if conv_state is not None:
            self.state["conv_state"] = conv_state
        if ffn_state is not None:
            self.state["ffn_state"] = ffn_state

        return self.state


class MiniCPMSALACache(DynamicCache):
    def __init__(self, config, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__(config=config)
        self.mixer_type = config.mixer_types
        if self.mixer_type[0] != "minicpm4":
            raise ValueError("The first layer must be 'minicpm4' to track seen tokens.")
        self.layers = (
            [
                (
                    InfLLMv2CacheLayer()
                    if self.mixer_type[index] == "minicpm4"
                    else LightningCacheLayer()
                )
                for index in range(num_hidden_layers)
            ]
            if num_hidden_layers
            else []
        )
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def update_no_rope_key(self, key_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update_no_rope_key(key_states)

    def update_compress_k(
        self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_compress_k(key_states, cu_seqlens)

    def update_no_compress_k(
        self, key_states, layer_idx, kernel_size=32, kernel_stride=16, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_no_compress_k(
            key_states, kernel_size, kernel_stride
        )

    def update_compress_k2(
        self, key_states, layer_idx, cu_seqlens=None, cache_kwargs=None
    ):
        return self.layers[layer_idx].update_compress_k2(key_states, cu_seqlens)

    def update_no_compress_k2(
        self,
        key_states,
        layer_idx,
        kernel_size=128,
        kernel_stride=64,
        cache_kwargs=None,
    ):
        return self.layers[layer_idx].update_no_compress_k2(
            key_states, kernel_size, kernel_stride
        )

    def crop(self, max_length):
        for layer in self.layers:
            layer.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)


# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def rms_layernorm(hidden: torch.Tensor, weight: torch.Tensor, eps: float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


class MiniCPMRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniCPMRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)


ALL_LAYERNORM_LAYERS.append(MiniCPMRMSNorm)


class MiniCPMRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.float32,
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MiniCPMLongRoPE(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        short_factor=None,
        long_factor=None,
        original_max_position_embeddings=None,
    ):
        self.short_factor = short_factor
        self.long_factor = long_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        scale = max_position_embeddings / self.original_max_position_embeddings
        self.scaling_factor = math.sqrt(
            1 + math.log(scale) / math.log(self.original_max_position_embeddings)
        )
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        if seq_len > self.original_max_position_embeddings:
            ext_factors = torch.tensor(
                self.long_factor, dtype=torch.float32, device=device
            )
        else:
            ext_factors = torch.tensor(
                self.short_factor, dtype=torch.float32, device=device
            )

        freqs = torch.mul(
            torch.outer(t, 1.0 / ext_factors).to(device=device),
            self.inv_freq.to(device=device).to(dtype),
        )
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype) * self.scaling_factor, persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype) * self.scaling_factor, persistent=False
        )


class MiniCPMLinearScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class MiniCPMDynamicNTKScalingRotaryEmbedding(MiniCPMRotaryEmbedding):
    """MiniCPMRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    orig_dtype = k.dtype
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # [bs, 1, seq_len, dim]
    q_fp32 = q.to(dtype=torch.float32, device=q.device)
    k_fp32 = k.to(dtype=torch.float32, device=k.device)
    q_embed = (q_fp32 * cos) + (rotate_half(q_fp32) * sin)
    k_embed = (k_fp32 * cos) + (rotate_half(k_fp32) * sin)
    return q_embed.to(dtype=orig_dtype), k_embed.to(dtype=orig_dtype)


class MiniCPMMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def _unpad_one_tensor(hidden_states, attention_mask):
    # Unpad the hidden states using the indices
    indices, cu_seqlens, max_seqlen_in_batch = _get_unpad_data(attention_mask)
    batch_size, seq_len = hidden_states.shape[:2]

    # Get the remaining dimensions
    remaining_dims = hidden_states.shape[2:]

    # Reshape to (batch_size * seq_len, *remaining_dims)
    reshaped_states = hidden_states.reshape(batch_size * seq_len, *remaining_dims)

    # Apply unpadding using indices
    unpadded_states = index_first_axis(reshaped_states, indices)

    return unpadded_states, indices, cu_seqlens, max_seqlen_in_batch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MiniCPMAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MiniCPMSALAConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        self._init_rope()

        # gated attn
        self.use_output_gate = config.attn_use_output_gate

        if self.use_output_gate:
            self.o_gate = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=config.attention_bias,
            )

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["rope_type"]
            scaling_factor = self.config.rope_scaling.get("factor", None)
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    short_factor=self.config.rope_scaling["short_factor"],
                    long_factor=self.config.rope_scaling["long_factor"],
                    base=self.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = position_ids.max().item() + 1
        cos, sin = None, None
        if self.config.attn_use_rope:
            cos, sin = self.rotary_emb(value_states.to(torch.float32), seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if os.getenv("MINICPM_SALA_FORCE_ALL_LIGHTNING"):
            # Use Simple GLA (same as LightningAttention) so HF and InfiniLM match.
            # Run in model dtype (bf16) to align with InfiniLM practice.
            attn_weights = None  # GLA does not expose attention weights
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_dtype = hidden_states.dtype
            q = query_states.transpose(1, 2).to(attn_dtype)  # (bsz, q_len, num_heads, head_dim)
            k = key_states.transpose(1, 2).to(attn_dtype)   # (bsz, kv_seq_len, num_heads, head_dim)
            v = value_states.transpose(1, 2).to(attn_dtype)
            decay = _build_slope_tensor(self.num_heads).to(q.device, dtype=attn_dtype) * (-1.0)
            scale = self.head_dim ** (-0.5)
            initial_state = None
            if past_key_value is not None and hasattr(past_key_value, "layers"):
                layer = past_key_value.layers[self.layer_idx]
                layer_state = getattr(layer, "state", None)
                if layer_state is not None:
                    initial_state = layer_state.get("recurrent_state", None)
            cu_seqlens = None
            if attention_mask is not None and bsz > 1:
                indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
                q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices).unsqueeze(0)
                k = index_first_axis(rearrange(k, "b s ... -> (b s) ..."), indices).unsqueeze(0)
                v = index_first_axis(rearrange(v, "b s ... -> (b s) ..."), indices).unsqueeze(0)
            if self.layer_idx in (0, 1) and os.getenv("INFINI_DEBUG_ATTN_DUMP"):
                try:
                    torch.save(q.detach().float().cpu(), f"/tmp/hf_layer{self.layer_idx}_q.pt")
                    torch.save(k.detach().float().cpu(), f"/tmp/hf_layer{self.layer_idx}_k.pt")
                    torch.save(v.detach().float().cpu(), f"/tmp/hf_layer{self.layer_idx}_v.pt")
                except Exception:
                    pass
            mode = "fused_recurrent" if q_len < 64 else "chunk"
            if mode == "chunk":
                o, final_state = chunk_simple_gla(
                    q=q, k=k, v=v,
                    g_gamma=decay,
                    initial_state=initial_state,
                    output_final_state=True,
                    scale=scale,
                    head_first=False,
                    cu_seqlens=cu_seqlens,
                )
            else:
                o, final_state = fused_recurrent_simple_gla(
                    q=q, k=k, v=v,
                    g_gamma=decay,
                    scale=scale,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                )
            if attention_mask is not None and bsz > 1:
                o = pad_input(o.squeeze(0), indices, bsz, q_len)
            attn_output = (
                rearrange(o, "b t h d -> b t (h d)").contiguous().to(hidden_states.dtype)
            )
            if past_key_value is not None and hasattr(past_key_value, "layers"):
                layer = past_key_value.layers[self.layer_idx]
                if getattr(layer, "state", None) is not None:
                    layer.update(
                        recurrent_state=final_state,
                        layer_idx=self.layer_idx,
                        offset=kv_seq_len,
                    )
            if self.layer_idx < 2 and os.getenv("INFINI_DEBUG_LOG"):
                import json
                import time

                def _log_tensor_stats(tensor, hypothesis_id, location, message):
                    t = tensor.detach().float().cpu()
                    payload = {
                        "sessionId": "9146ea",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": {
                            "layer": int(self.layer_idx),
                            "shape": list(t.shape),
                            "min": float(t.min().item()),
                            "max": float(t.max().item()),
                            "mean": float(t.mean().item()),
                            "l2": float(t.norm().item()),
                        },
                        "timestamp": int(time.time() * 1000),
                    }
                    with open(os.environ["INFINI_DEBUG_LOG"], "a") as f:
                        f.write(json.dumps(payload) + "\n")

                _log_tensor_stats(
                    attn_output,
                    "HF_A",
                    "modeling_minicpm_sala.py:hf_attn_pre_gate",
                    "HF attn pre-gate",
                )
                if self.layer_idx in (0, 1):
                    torch.save(attn_output.detach().float().cpu(), f"/tmp/hf_attn_out_layer{self.layer_idx}.pt")
            if self.use_output_gate:
                o_gate = self.o_gate(hidden_states)
                attn_output = attn_output * F.sigmoid(o_gate)
            attn_output = self.o_proj(attn_output)
            if self.layer_idx < 2 and os.getenv("INFINI_DEBUG_LOG"):
                import json
                import time

                t = attn_output.detach().float().cpu()
                payload = {
                    "sessionId": "9146ea",
                    "hypothesisId": "HF_A",
                    "location": "modeling_minicpm_sala.py:hf_attn_post_oproj",
                    "message": "HF attn post-o_proj",
                    "data": {
                        "layer": int(self.layer_idx),
                        "shape": list(t.shape),
                        "min": float(t.min().item()),
                        "max": float(t.max().item()),
                        "mean": float(t.mean().item()),
                        "l2": float(t.norm().item()),
                    },
                    "timestamp": int(time.time() * 1000),
                }
                with open(os.environ["INFINI_DEBUG_LOG"], "a") as f:
                    f.write(json.dumps(payload) + "\n")

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        # gated attn
        if self.use_output_gate:
            o_gate = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(o_gate)
        # gated attn

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniCPMFlashAttention2(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # MiniCPMFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = position_ids.max().item() + 1
        cos, sin = None, None
        if self.config.attn_use_rope:
            cos, sin = self.rotary_emb(
                value_states.to(torch.float32), seq_len=kv_seq_len
            )
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (MiniCPMRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        if self.use_output_gate:
            o_gate = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(o_gate)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MiniCPMFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MiniCPMInfLLMv2Attention(MiniCPMAttention):
    """
    MiniCPM flash attention module. This module inherits from `MiniCPMAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.config._attn_implementation == "flash_attention_2"
        ), "Only flash_attention_2 is supported for sparse attention"
        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

        #  -------sparse-------
        self.kernel_size = self.config.sparse_config.get("kernel_size", 32)
        self.kernel_stride = self.config.sparse_config.get("kernel_stride", 16)
        self.init_blocks = self.config.sparse_config.get("init_blocks", 1)
        self.block_size = self.config.sparse_config.get("block_size", 64)
        self.window_size = self.config.sparse_config.get("window_size", 2048)
        self.dense_len = self.config.sparse_config.get("dense_len", 8192)

        self.local_blocks = self.window_size // self.block_size  # local_blocks
        self.topk = self.config.sparse_config.get("topk", 64) + (
            self.window_size // self.block_size
        )
        self.use_nope = self.config.sparse_config.get("use_nope", False)

        self.compress_k = CompressK(
            self.num_key_value_heads,
            self.head_dim,
            kernel_size=self.kernel_size,
            kernel_stride=self.kernel_stride,
        )
        self.compress_k2 = CompressK(
            self.num_key_value_heads,
            self.head_dim,
            kernel_size=self.kernel_size * 4,
            kernel_stride=self.kernel_stride * 4,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # MiniCPMFlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.use_nope:
            query_states_no_rope = query_states.view(
                bsz, q_len, self.num_heads, self.head_dim
            )
            key_states_no_rope = key_states.view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = position_ids.max().item() + 1
        cos, sin = None, None
        if self.config.attn_use_rope:
            cos, sin = self.rotary_emb(
                value_states.to(torch.float32), seq_len=kv_seq_len
            )
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if self.use_nope:
            key_states_no_rope = past_key_value.update_no_rope_key(
                key_states_no_rope, self.layer_idx
            )
            no_rope_param = {
                "key_states_no_rope": key_states_no_rope,
                "query_states_no_rope": query_states_no_rope,
            }

        else:
            no_rope_param = None

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (MiniCPMRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        if kv_seq_len < self.dense_len:
            attn_output = self._flash_attention_forward_dense(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
            )
        else:
            attn_output = self._sparse_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                no_rope_param=no_rope_param,  # if past_key_value is not None else None,
                past_key_value=past_key_value,
            )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        if self.use_output_gate:
            o_gate = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(o_gate)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _sparse_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        no_rope_param=None,
        past_key_value=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MiniCPMFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            if past_key_value is not None:
                (
                    compressed_k,
                    compressed_cu_seqlens,
                    compressed_k2,
                    compressed_cu_seqlens2,
                ) = self.get_compress_k(
                    key_states=(
                        key_states
                        if self.use_nope == False
                        else no_rope_param["key_states_no_rope"]
                    ),  # This can be optimized a bit;
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                )

            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            if no_rope_param is not None:
                if max_seqlen_in_batch_q == 1:
                    no_rope_param["query_states_no_rope"] = no_rope_param[
                        "query_states_no_rope"
                    ].squeeze(1)
                else:
                    no_rope_param["query_states_no_rope"], _, _, _ = _unpad_one_tensor(
                        no_rope_param["query_states_no_rope"],
                        attention_mask=attention_mask,
                    )
            if past_key_value is None:
                # compress_k use varlen form
                compressed_k, compressed_cu_seqlens = self.compress_k(
                    key_states, cu_seqlens_k
                )
                compressed_k2, compressed_cu_seqlens2 = self.compress_k2(
                    key_states, cu_seqlens_k
                )
            else:
                # compressed_k and compressed_k2 already retrieved from get_compress_k above
                pass

            attn_output_unpad = self.sparse_forward(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                no_rope_param=no_rope_param,
                compressed_k=compressed_k,
                compressed_cu_seqlens=compressed_cu_seqlens,
                compressed_k2=compressed_k2,
                compressed_cu_seqlens2=compressed_cu_seqlens2,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )

        else:
            raise ValueError("Need attention mask")

        return attn_output

    def get_compress_k(self, key_states, attention_mask, past_key_value):
        """
        Get compressed key states and corresponding cumulative sequence lengths.

        Args:
            key_states: Key states tensor
            cu_seqlens_k: Cumulative sequence lengths for keys
            past_key_value: Past key-value cache
            no_rope_param: Optional parameter containing key states without rope

        Returns:
            Tuple of (compressed_k, compressed_cu_seqlens, compressed_k2, compressed_cu_seqlens2)
        """

        # Check if this is prefilling or initial compression condition

        is_prefilling = key_states.shape[1] >= self.dense_len and (
            not past_key_value.layers[self.layer_idx].compress_k_cache
        )

        if is_prefilling:
            unpadded_key_states, indices, cu_seqlens, max_seqlen_in_batch = (
                _unpad_one_tensor(key_states, attention_mask=attention_mask)
            )
            # Compress the keys
            compressed_k, compressed_cu_seqlens = self.compress_k(
                unpadded_key_states, cu_seqlens
            )
            compressed_k2, compressed_cu_seqlens2 = self.compress_k2(
                unpadded_key_states, cu_seqlens
            )

            past_key_value.update_compress_k(
                compressed_k, self.layer_idx, compressed_cu_seqlens
            )
            past_key_value.update_compress_k2(
                compressed_k2, self.layer_idx, compressed_cu_seqlens2
            )

            no_compress_k_list = []
            # Compute and update no_compress_k
            for i in range(len(compressed_cu_seqlens) - 1):
                no_compress_k_start = (
                    compressed_cu_seqlens[i + 1] - compressed_cu_seqlens[i]
                ) * self.kernel_stride

                no_compress_k_list.append(
                    unpadded_key_states[
                        cu_seqlens[i] + no_compress_k_start : cu_seqlens[i + 1]
                    ].clone()
                )

            past_key_value.update_no_compress_k(
                no_compress_k_list,
                self.layer_idx,
                kernel_stride=self.kernel_stride,
                kernel_size=self.kernel_size,
            )

            # Also update no_compress_k2
            no_compress_k2_list = []
            for i in range(len(compressed_cu_seqlens2) - 1):
                no_compress_k2_start = (
                    (compressed_cu_seqlens2[i + 1] - compressed_cu_seqlens2[i])
                    * self.kernel_stride
                    * 4
                )

                no_compress_k2_list.append(
                    unpadded_key_states[
                        cu_seqlens[i] + no_compress_k2_start : cu_seqlens[i + 1]
                    ].clone()
                )

            past_key_value.update_no_compress_k2(
                no_compress_k2_list,
                self.layer_idx,
                kernel_stride=self.kernel_stride * 4,
                kernel_size=self.kernel_size * 4,
            )

        else:
            # Decode case: incremental update
            batch_size = key_states.shape[
                0
            ]  # key_states.shape = [batch_size, seq, k_head_num, head_dim]
            key_states_split = list(
                torch.split(
                    key_states[:, -1:].squeeze(
                        1
                    ),  # [batch_size, seq, k_head_num, head_dim]->[batch_size, 1, k_head_num, head_dim]-> [batch_size, k_head_num, head_dim]
                    [1] * batch_size,
                    dim=0,
                )
            )
            # Try to update no_compress_k buffer
            no_compress_k_list = past_key_value.update_no_compress_k(
                key_states_split,
                self.layer_idx,
                kernel_stride=self.kernel_stride,
                kernel_size=self.kernel_size,
            )
            new_compressed_k_list = []
            for no_compress_k in no_compress_k_list:

                if no_compress_k is not None:
                    # We have enough tokens to compress
                    new_compressed_k = no_compress_k.mean(
                        dim=0, keepdim=True
                    )  # [1, n_heads_k, head_dim]

                    new_compressed_k_list.append(new_compressed_k)
                else:
                    new_compressed_k_list.append(None)
            compressed_k, compressed_cu_seqlens = past_key_value.update_compress_k(
                new_compressed_k_list,
                self.layer_idx,
            )

            # For compress_k2, update no_compress_k2 buffer and compress when ready
            no_compress_k2_list = past_key_value.update_no_compress_k2(
                key_states_split,
                self.layer_idx,
                kernel_stride=self.kernel_stride * 4,
                kernel_size=self.kernel_size * 4,
            )
            new_compressed_k2_list = []
            for no_compress_k2 in no_compress_k2_list:
                if no_compress_k2 is not None:
                    # We have enough tokens to compress for k2
                    new_compressed_k2 = no_compress_k2.mean(
                        dim=0, keepdim=True
                    )  # [1, n_heads_k, head_dim]
                    new_compressed_k2_list.append(new_compressed_k2)
                else:
                    new_compressed_k2_list.append(None)
            compressed_k2, compressed_cu_seqlens2 = past_key_value.update_compress_k2(
                new_compressed_k2_list,
                self.layer_idx,
            )

        return (
            compressed_k,
            compressed_cu_seqlens,
            compressed_k2,
            compressed_cu_seqlens2,
        )

    def sparse_forward(
        self,
        query_layer,
        key_layer,
        value_layer,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        no_rope_param=None,
        compressed_k=None,
        compressed_cu_seqlens=None,
        compressed_k2=None,
        compressed_cu_seqlens2=None,
    ):
        # Handle q_head/k_head ratio for infllmv2 (requires 16:1 ratio)
        num_q_heads = query_layer.shape[-2]
        num_k_heads = key_layer.shape[-2]
        current_ratio = num_q_heads // num_k_heads
        required_ratio = 16

        if current_ratio < required_ratio:
            repeat_times = required_ratio // current_ratio
            query_layer = query_layer.repeat_interleave(repeat_times, dim=-2)
        else:
            repeat_times = 1
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        cache_lens = None
        if max_seqlen_in_batch_q == 1 and max_seqlen_in_batch_k > 1:  # decoding
            seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
            cache_lens = seq_lens_k - 1

        topk_idx = compressed_attention(
            (
                query_layer
                if no_rope_param is None
                else no_rope_param["query_states_no_rope"]
            ),
            compressed_k,
            compressed_k2,
            self.kernel_size,
            self.kernel_stride,
            self.block_size,
            self.topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            compressed_cu_seqlens2,
            max_seqlen_in_batch_q,
            compressed_seqlens.max().item(),
            None,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            cache_lens=cache_lens,
        )
        topk_attn_output = infllmv2_attn_varlen_func(
            query_layer,
            key_layer,
            value_layer,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_in_batch_q,
            max_seqlen_in_batch_k,
            dropout_p=0.0,
            deterministic=False,
            softmax_scale=None,
            causal=max_seqlen_in_batch_q != 1,
            return_attn_probs=False,
            topk_idx=topk_idx,
        )
        if repeat_times > 1:
            topk_attn_output = topk_attn_output.view(
                topk_attn_output.shape[0],
                topk_attn_output.shape[-2] // repeat_times,
                repeat_times,
                -1,
            ).mean(dim=-2)
        return topk_attn_output

    def _flash_attention_forward_dense(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in MiniCPMFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


def index_first_axis(x, indices):
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, "b ... -> b (...)"),
        0,
        repeat(indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)


def index_put_first_axis(x, indices, first_axis_dim):
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    # TODO [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
    y[indices] = x
    # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
    return y


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    q: torch.Tensor,
    states: tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, seq_len, *_ = states[0].shape

    state = tuple(
        index_first_axis(rearrange(s, "b s ... -> (b s) ..."), indices_k)
        for s in states
    )

    if q_len == seq_len:
        q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError(
            "We only support either q_len == k_len (prefilling) or q_len == 1 (decoding)"
        )

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return (
        q,
        state,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


def _build_slope_tensor(nheads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(nheads))  # (nheads,)
    return slopes


class LightningAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: MiniCPMSALAConfig,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_dropout: float = 0.0,
        use_output_gate: bool = False,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        use_rope: bool = False,
        use_output_norm: bool = False,
        qk_norm: bool = True,
        rope_head_dim: Optional[int] = None,
        scale: str = "1/sqrt(d)",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        if scale == "1/sqrt(d)":
            self.scale = self.head_dim ** (-0.5)
        elif scale == "1/d":
            self.scale = self.head_dim ** (-1.0)
        else:
            self.scale = 1.0
        self.attention_dropout = attention_dropout
        self.is_causal = True
        self.use_output_gate = use_output_gate
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.use_output_norm = use_output_norm
        self.rope_head_dim = rope_head_dim if rope_head_dim is not None else head_dim
        assert self.rope_head_dim <= self.head_dim

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=self.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=self.attention_bias,
        )
        if self.use_output_norm:
            self.o_norm = MiniCPMRMSNorm(
                self.num_attention_heads * self.head_dim, eps=self.rms_norm_eps
            )

        if self.use_output_gate:
            self.z_proj = nn.Linear(
                self.hidden_size,
                self.num_attention_heads * self.head_dim,
                bias=self.attention_bias,
            )

        if self.qk_norm:
            self.q_norm = MiniCPMRMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.k_norm = MiniCPMRMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = MiniCPMRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                base=self.config.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["rope_type"]
            scaling_factor = self.config.rope_scaling.get("factor", None)
            if scaling_type == "linear":
                self.rotary_emb = MiniCPMLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.config.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = MiniCPMDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.config.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.config.rope_theta,
                )
            elif scaling_type == "longrope":
                self.rotary_emb = MiniCPMLongRoPE(
                    self.head_dim,
                    max_position_embeddings=self.config.max_position_embeddings,
                    short_factor=self.config.rope_scaling["short_factor"],
                    long_factor=self.config.rope_scaling["long_factor"],
                    base=self.config.rope_theta,
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def attn_fn(
        self,
        q: Tensor,  # (b, t, h, d)
        k: Tensor,  # (b, t, h, d)
        v: Tensor,  # (b, t, h, d)
        decay: Tensor,  # (h,)
        scale: float | None = None,  # will use dk^(-1) if None.
        initial_state: Tensor | None = None,  # (b, h, dk, dv)
        mode: str = "chunk",
        attention_mask=None,
    ) -> tuple[Tensor, Tensor]:
        seqlen = q.shape[1]
        q_len = q.shape[1]
        mode = "fused_recurrent" if seqlen < 64 else "chunk"
        batch_size = q.shape[0]
        cu_seqlens = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            q = index_first_axis(
                rearrange(q, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)
            k = index_first_axis(
                rearrange(k, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)
            v = index_first_axis(
                rearrange(v, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)
        elif batch_size > 1:
            raise ValueError("attention_mask must be provided when batch size > 1")
        if mode == "chunk":
            o, final_state = chunk_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=decay,  # (h,)
                initial_state=initial_state,
                output_final_state=True,
                scale=scale,
                head_first=False,
                cu_seqlens=cu_seqlens,
            )  # (b, t, h, d)
        elif mode == "fused_recurrent":
            o, final_state = fused_recurrent_simple_gla(
                q=q,
                k=k,
                v=v,
                g_gamma=decay,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, final_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, seqlen, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = rearrange(q, "b t (h d) -> b h t d", d=self.head_dim)
        k = rearrange(k, "b t (h d) -> b h t d", d=self.head_dim)
        v = rearrange(v, "b t (h d) -> b h t d", d=self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            kv_seq_len = position_ids.max().item() + 1
            cos, sin = self.rotary_emb(v.to(torch.float32), seq_len=kv_seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        s = _build_slope_tensor(self.num_attention_heads).to(
            k.device, dtype=torch.float32
        ) * (
            -1.0
        )  # (h)

        initial_state = None
        if past_key_value is not None:
            layer_state = past_key_value.layers[self.layer_idx].state
            initial_state = layer_state.get("recurrent_state", None)

        q = rearrange(q, "b h t d -> b t h d").to(torch.float32)
        k = rearrange(k, "b h t d -> b t h d").to(torch.float32)
        v = rearrange(v, "b h t d -> b t h d").to(torch.float32)
        s = s.to(torch.float32)

        if self.layer_idx == 1 and os.getenv("INFINI_DEBUG_ATTN_DUMP"):
            try:
                torch.save(q.detach().float().cpu(), "/tmp/hf_layer1_q.pt")
                torch.save(k.detach().float().cpu(), "/tmp/hf_layer1_k.pt")
                torch.save(v.detach().float().cpu(), "/tmp/hf_layer1_v.pt")
            except Exception:
                pass

        o, final_state = self.attn_fn(
            q=q,
            k=k,
            v=v,
            decay=s,
            initial_state=initial_state,
            scale=self.scale,
            attention_mask=attention_mask,
        )

        if past_key_value is not None:
            past_key_value.layers[self.layer_idx].update(
                recurrent_state=final_state,
                layer_idx=self.layer_idx,
                offset=seqlen,
            )

        o = (
            rearrange(o, "b t h d -> b t (h d)").contiguous().to(hidden_states.dtype)
        )  # (b, t, d)

        if self.layer_idx == 1 and os.getenv("INFINI_DEBUG_ATTN_DUMP"):
            try:
                torch.save(o.detach().float().cpu(), "/tmp/hf_attn_out_layer1.pt")
            except Exception:
                pass

        if self.use_output_norm:
            o = self.o_norm(o)  # (b, t, d)

        if self.use_output_gate:
            z = F.sigmoid(self.z_proj(hidden_states))  # (b, t, d)
            o = o * z  # (b, t, d)

        y = self.o_proj(o)
        return y, None, past_key_value


class MiniCPMSdpaAttention(MiniCPMAttention):
    """
    MiniCPM attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `MiniCPMAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from MiniCPMAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "MiniCPMSALAModel is using MiniCPMSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = position_ids.max().item() + 1
        cos, sin = None, None
        if self.config.attn_use_rope:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.use_output_gate:
            o_gate = self.o_gate(hidden_states)
            attn_output = attn_output * F.sigmoid(o_gate)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


MINICPM_ATTENTION_CLASSES = {
    "eager": MiniCPMAttention,
    "flash_attention_2": MiniCPMFlashAttention2,
    "sdpa": MiniCPMSdpaAttention,
}


class MiniCPMSALADecoderLayer(nn.Module):
    def __init__(self, config: MiniCPMSALAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.mixer_type = config.mixer_types[layer_idx]
        if self.mixer_type == "minicpm4":
            if os.getenv("MINICPM_SALA_FORCE_ALL_LIGHTNING"):
                self.self_attn = MiniCPMAttention(config=config, layer_idx=layer_idx)
            else:
                self.self_attn = MINICPM_ATTENTION_CLASSES[
                    config._attn_implementation
                ](config=config, layer_idx=layer_idx)
        elif self.mixer_type in ["lightning", "lightning_attn", "lightning-attn"]:
            assert (
                config.head_dim is not None
            ), "head_dim must be provided for LightningAttention"
            self.self_attn = LightningAttention(
                config=config,
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.lightning_nkv,
                head_dim=config.head_dim,
                attention_dropout=config.attention_dropout,
                use_output_gate=config.use_output_gate,
                attention_bias=config.attention_bias,
                rms_norm_eps=config.rms_norm_eps,
                use_rope=config.lightning_use_rope,
                qk_norm=config.qk_norm,
                use_output_norm=config.use_output_norm,
                scale=config.lightning_scale,
            )
        else:
            raise ValueError(f"Unsupported mixer type: {self.mixer_type}")

        self.mlp = MiniCPMMLP(config)
        self.input_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiniCPMRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.scale_depth = config.scale_depth
        self.num_hidden_layers = config.num_hidden_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_idx < 2 and os.getenv("INFINI_DEBUG_LOG"):
            import json
            import time

            t = hidden_states.detach().float().cpu()
            payload = {
                "sessionId": "9146ea",
                "hypothesisId": "HF_B",
                "location": "modeling_minicpm_sala.py:hf_input_layernorm",
                "message": "HF input layernorm output",
                "data": {
                    "layer": int(self.layer_idx),
                    "shape": list(t.shape),
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                    "mean": float(t.mean().item()),
                    "l2": float(t.norm().item()),
                },
                "timestamp": int(time.time() * 1000),
            }
            with open(os.environ["INFINI_DEBUG_LOG"], "a") as f:
                f.write(json.dumps(payload) + "\n")
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = residual + hidden_states * (
            self.scale_depth / math.sqrt(self.num_hidden_layers)
        )

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * (
            self.scale_depth / math.sqrt(self.num_hidden_layers)
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


MINICPM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MiniCPMSALAConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMSALAPreTrainedModel(PreTrainedModel):
    config_class = MiniCPMSALAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiniCPMSALADecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


MINICPM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MiniCPM Model outputting raw hidden-states without any specific head on top.",
    MINICPM_START_DOCSTRING,
)
class MiniCPMSALAModel(MiniCPMSALAPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiniCPMSALADecoderLayer`]

    Args:
        config: MiniCPMSALAConfig
    """

    def __init__(self, config: MiniCPMSALAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                MiniCPMSALADecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.norm = MiniCPMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            # Calculate the usable length of past key values
            past_key_values_length = (
                past_key_values.get_seq_length()
                if isinstance(past_key_values, MiniCPMSALACache)
                else 0
            )

            # Initialize MiniCPMSALACache if needed
            if (
                self.config.sparse_config is not None
                and torch.cuda.is_available()
                and past_key_values_length == 0
            ):
                past_key_values = MiniCPMSALACache(
                    config=self.config, num_hidden_layers=self.config.num_hidden_layers
                )

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.config.scale_emb
        if os.getenv("INFINI_DEBUG_LOG"):
            import json
            import time

            t = inputs_embeds.detach().float().cpu()
            payload = {
                "sessionId": "9146ea",
                "hypothesisId": "HF_C",
                "location": "modeling_minicpm_sala.py:hf_inputs_embeds",
                "message": "HF inputs_embeds",
                "data": {
                    "shape": list(t.shape),
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                    "mean": float(t.mean().item()),
                    "l2": float(t.norm().item()),
                },
                "timestamp": int(time.time() * 1000),
            }
            with open(os.environ["INFINI_DEBUG_LOG"], "a") as f:
                f.write(json.dumps(payload) + "\n")

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            pass
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiniCPMSALAForCausalLM(MiniCPMSALAPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniCPMSALAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MiniCPMSALAForCausalLM

        >>> model = MiniCPMSALAForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        hidden_states = hidden_states[:, slice_indices, :].contiguous()
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(
                hidden_states / (self.config.hidden_size / self.config.dim_model_base)
            )
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                # Use the new Cache class methods
                cache_length = past_key_values.get_seq_length()

                if torch.cuda.is_available() and cache_length == 0:
                    past_key_values = MiniCPMSALACache(
                        config=self.config,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )
                past_length = cache_length
                max_cache_length = None
            else:
                raise ValueError(
                    "You must use the new past_key_values format, such as the Cache class, instead of the old tuple format."
                )

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        # Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    @torch.inference_mode()
    def chat(
        self,
        tokenizer,
        query: str,
        history: List[Dict] = None,
        role: str = "user",
        max_length: int = 4096,
        num_beams=1,
        do_sample=True,
        top_p=0.8,
        temperature=0.3,
        logits_processor=None,
        **kwargs,
    ):
        if history is None:
            history = []
        if logits_processor:
            gen_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": temperature,
                "logits_processor": logits_processor,
                **kwargs,
            }
        else:
            gen_kwargs = {
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "top_p": top_p,
                "temperature": temperature,
                "logits_processor": logits_processor,
                **kwargs,
            }

        history.append({"role": role, "content": query})
        history_str = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(history_str, return_tensors="pt").to(self.device)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) : -1]
        response = tokenizer.decode(outputs)
        pattern = re.compile(r".*?(?=<AI>|<用户>)", re.DOTALL)
        matches = pattern.findall(response)
        if len(matches) > 0:
            response = matches[0]
        history.append({"role": "assistant", "content": response})
        return response, history


@add_start_docstrings(
    """
    The MiniCPM Model transformer with a sequence classification head on top (linear layer).

    [`MiniCPMSALAForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MINICPM_START_DOCSTRING,
)
class MiniCPMSALAForSequenceClassification(MiniCPMSALAPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MiniCPMSALAModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(MINICPM_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
