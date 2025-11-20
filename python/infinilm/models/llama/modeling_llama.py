# Copyright (c) 2025, InfiniCore
#
# This file contains modified code derived from transformers
# implementation, which is licensed under the BSD 3-Clause License.
#
# The modifications include adaptations for the InfiniCore framework.
#
# Original transformers source:
# https://github.com/huggingface/transformers
#
# Referencing PyTorch v4.57.0
#
# The use of this file is governed by the BSD 3-Clause License.


import json
import os
from typing import Optional, Union

from transformers.utils import logging

import infinicore

from ...cache_utils import Cache, DynamicCache
from ...generation.utils import GenerationMixin
from .configuration_llama import LlamaConfig

logger = logging.get_logger(__name__)


def repeat_kv(keys: infinicore.Tensor, values: infinicore.Tensor, ngroup: int):
    total_seq_len, num_key_value_heads, head_dim = keys.shape

    keys_repeat = infinicore.empty(
        (total_seq_len, num_key_value_heads, ngroup, head_dim),
        dtype=keys.dtype,
        device=keys.device,
    )
    values_repeat = infinicore.empty(
        (total_seq_len, num_key_value_heads, ngroup, head_dim),
        dtype=values.dtype,
        device=values.device,
    )

    for i in range(ngroup):
        keys_repeat.narrow(2, i, 1).copy_(
            keys.view((total_seq_len, num_key_value_heads, 1, head_dim))
        )
        values_repeat.narrow(2, i, 1).copy_(
            values.view((total_seq_len, num_key_value_heads, 1, head_dim))
        )

    keys_new = keys_repeat.view((total_seq_len, num_key_value_heads * ngroup, head_dim))
    values_new = values_repeat.view(
        (total_seq_len, num_key_value_heads * ngroup, head_dim)
    )
    return keys_new, values_new


def multi_head_attention(
    querys: infinicore.Tensor,  # [seq_len,       num_heads, head_dim]
    keys: infinicore.Tensor,  #   [total_seq_len, num_heads, head_dim]
    values: infinicore.Tensor,  # [total_seq_len, num_heads, head_dim]
    scaling: float,
):
    # => [ num_heads, seq_len,       head_dim]
    Q = querys.permute((1, 0, 2))
    # => [ num_heads, total_seq_len, head_dim]
    K = keys
    # => [ num_heads, total_seq_len, head_dim]
    V = values.permute((1, 0, 2))

    # [num_heads, seq_len, head_dim] @ [ num_heads, head_dim, total_seq_len]
    # => [ num_heads, seq_len, total_seq_len]
    attn_weight = Q @ K.permute((1, 2, 0))

    scaling = infinicore.from_list(
        [scaling], dtype=attn_weight.dtype, device=attn_weight.device
    ).as_strided(attn_weight.shape, [0, 0, 0])

    attn_weight = attn_weight * scaling

    infinicore.nn.functional.causal_softmax(attn_weight, out=attn_weight)

    # [ num_heads,  seq_len,  total_seq_len] @ [num_heads, total_seq_len, head_dim]
    # => [ num_heads,seq_len,head_dim]
    out = attn_weight @ V

    # => [seq_len, num_heads, head_dim]
    return out.permute((1, 0, 2)).contiguous()


def grouped_query_attention(
    querys: infinicore.Tensor,  # [seq_len,       num_attention_heads, head_dim]
    keys: infinicore.Tensor,  #   [total_seq_len, num_key_value_heads, head_dim]
    values: infinicore.Tensor,  # [total_seq_len, num_key_value_heads, head_dim]
    scaling: float,
):
    num_attention_heads = querys.shape[1]
    num_key_value_heads = keys.shape[1]
    ngroup = num_attention_heads // num_key_value_heads
    if ngroup > 1:
        keys, values = repeat_kv(keys, values, ngroup)

    return multi_head_attention(querys, keys, values, scaling=scaling)


LlamaRMSNorm = infinicore.nn.RMSNorm


class LlamaMLP(infinicore.nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        mlp_bias = config.mlp_bias

        self.gate_proj = infinicore.nn.Linear(
            hidden_size, intermediate_size, bias=mlp_bias, **kwargs
        )
        self.up_proj = infinicore.nn.Linear(
            hidden_size, intermediate_size, bias=mlp_bias, **kwargs
        )
        self.down_proj = infinicore.nn.Linear(
            intermediate_size, hidden_size, bias=mlp_bias, **kwargs
        )
        self.act_fn = infinicore.nn.functional.silu

    def forward(self, x: infinicore.Tensor) -> infinicore.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaAttention(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        attention_bias = config.attention_bias

        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_attention_heads
        )

        self.scaling = self.head_dim**-0.5

        self.q_proj = infinicore.nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=attention_bias,
            **kwargs,
        )

        self.k_proj = infinicore.nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
            **kwargs,
        )

        self.v_proj = infinicore.nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
            **kwargs,
        )

        self.o_proj = infinicore.nn.Linear(
            self.num_attention_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: infinicore.Tensor,
        past_key_values: Optional[Cache] = None,
        rope_instance: infinicore.nn.RoPE = None,
        **kwargs,
    ) -> infinicore.Tensor:
        hidden_states_shape = hidden_states.shape  # [bs, seq_len, hidden_size]
        bs, seq_len = hidden_states_shape[:-1]  #    [bs, seq_len]

        querys_shape = (bs, seq_len, self.num_attention_heads, self.head_dim)
        keys_shape = (bs, seq_len, self.num_key_value_heads, self.head_dim)
        values_shape = (bs, seq_len, self.num_key_value_heads, self.head_dim)

        # --------------------------------------------------------------------------------------- #
        #                           对 Q,K，V进行 project
        # --------------------------------------------------------------------------------------- #
        # => [bs, seq_len,  num_attention_heads, head_dim]
        query_states = self.q_proj(hidden_states).view(querys_shape)

        # => [bs, seq_len,  num_key_value_heads, head_dim]
        key_states = self.k_proj(hidden_states).view(keys_shape)

        # => [bs, seq_len, nkvh, head_dim]
        value_states = self.v_proj(hidden_states).view(values_shape)

        # --------------------------------------------------------------------------------------- #
        #                           对 Q和K， 加上 rope
        # --------------------------------------------------------------------------------------- #
        cache_position = kwargs.pop("cache_position", None)
        if cache_position is None:
            raise KeyError("cache_position error")
        if rope_instance is None:
            raise KeyError("rope_instance error")

        query_states = rope_instance(query_states, cache_position)
        key_states = rope_instance(key_states, cache_position)

        # --------------------------------------------------------------------------------------- #
        #                           kv cache
        # --------------------------------------------------------------------------------------- #
        if past_key_values is not None:
            cache_kwargs = {}
            key_states_total, value_states_total = past_key_values.update(
                key_states,  # [bs, seq_len, num_key_value_heads, head_dim]
                value_states,  # [bs, seq_len, num_key_value_heads, head_dim]
                self.layer_idx,
                cache_kwargs,
            )

        # --------------------------------------------------------------------------------------- #
        #                           注意力计算
        # --------------------------------------------------------------------------------------- #
        total_seq_len = key_states_total.shape[1]
        attn_output = infinicore.empty_like(query_states)
        for i in range(0, bs):
            query_states_i = query_states.narrow(0, i, 1).view(
                (seq_len, self.num_attention_heads, self.head_dim)
            )
            key_states_i = key_states_total.narrow(0, i, 1).view(
                (total_seq_len, self.num_key_value_heads, self.head_dim)
            )
            value_states_i = value_states_total.narrow(0, i, 1).view(
                (total_seq_len, self.num_key_value_heads, self.head_dim)
            )

            attn_output_i = attn_output.narrow(0, i, 1).view(
                (seq_len, self.num_attention_heads, self.head_dim)
            )

            attention_i = grouped_query_attention(
                query_states_i, key_states_i, value_states_i, scaling=self.scaling
            )

            attn_output_i.copy_(attention_i)

        # --------------------------------------------------------------------------------------- #
        #                           out project
        # --------------------------------------------------------------------------------------- #
        # ([bs, seq_len, num_attention_heads, head_dim]) ==> [bs, seq_len, hidden_size ]
        attn_output = attn_output.view(hidden_states_shape)

        # o_proj
        return self.o_proj(attn_output)


class LlamaDecoderLayer(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, **kwargs):
        super().__init__()
        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, **kwargs)
        self.mlp = LlamaMLP(config=config, **kwargs)

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps, **kwargs)
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size, eps=rms_norm_eps, **kwargs
        )

    def forward(
        self,
        hidden_states: infinicore.Tensor,  # [bs, seq_len, hidden_size]
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        rope_instance=None,
        **kwargs,
    ) -> infinicore.Tensor:
        # ------------------------------------------------ #
        #          Self Attention
        # ------------------------------------------------ #
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            rope_instance=rope_instance,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # ------------------------------------------------ #
        #           Fully Connected
        # ------------------------------------------------ #
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(infinicore.nn.Module):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.embed_tokens = infinicore.nn.Embedding(
            config.vocab_size, config.hidden_size, **kwargs
        )

        self.layers = infinicore.nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx, **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, **kwargs)

        self.rope_instance = infinicore.nn.RoPE(
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            head_dim=head_dim,
            **kwargs,
        )

    def forward(
        self,
        input_ids,
        cache_position,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,  # True
        **kwargs,
    ):
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # --------------------------------------------------------- #
        #               token的embedding
        # --------------------------------------------------------- #
        # input_ids :     {1,5}       tensor([[    1,  1128,   526,   366, 29892]])
        # inputs_embeds : {1,5,2048}  tensor([[[...]]])
        inputs_embeds = self.embed_tokens(input_ids)

        # --------------------------------------------------------- #
        #                    decoder_layer
        # --------------------------------------------------------- #
        ilayer = 0  # noqa: F841
        hidden_states = inputs_embeds
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            # print("ilayer: ", ilayer)
            # ilayer += 1
            hidden_states = decoder_layer(
                hidden_states,
                past_key_values=past_key_values,
                cache_position=cache_position,
                rope_instance=self.rope_instance,
                **kwargs,
            )

        # --------------------------------------------------------- #
        #                    norm
        # --------------------------------------------------------- #
        seq_len = hidden_states.shape[1]
        last_token = hidden_states.narrow(1, seq_len - 1, 1)

        return self.norm(last_token)


class LlamaForCausalLM(infinicore.nn.Module, GenerationMixin):
    config: LlamaConfig

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, **kwargs)
        self.lm_head = infinicore.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            **kwargs,
        )

    def forward(
        self,
        input_ids,
        cache_position,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        last_token = self.model(
            input_ids,
            cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return self.lm_head(last_token)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Optional[Union[str, os.PathLike]],
        device: infinicore.device,
        dtype=infinicore.dtype,
    ):
        def load_config_json(dir_path_: str):
            with open(os.path.join(dir_path_, "config.json"), "r") as f:
                config = json.load(f)
            return config

        config_dict = load_config_json(os.path.join(model_path))
        config = LlamaConfig(**config_dict)

        return LlamaForCausalLM(config, device=device, dtype=dtype)


__all__ = [
    "LlamaModel",
    "LlamaForCausalLM",
]
