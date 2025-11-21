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


from abc import ABC, abstractmethod
from typing import Any, Optional

import transformers.utils.logging as logging

import infinicore

logger = logging.get_logger(__name__)


class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    def __init__(self):
        self.keys, self.values = None, None

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @abstractmethod
    def lazy_initialization(self, key_states: infinicore.Tensor): ...

    @abstractmethod
    def update(
        self,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[infinicore.Tensor, infinicore.Tensor]: ...


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated.
    It stores the key and value states as tensors of shape `[batch_size, seq_len, num_heads, head_dim]`.
    """

    def __init__(self, max_position_embeddings):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.cache_position = 0

    def lazy_initialization(self, key_states: infinicore.Tensor):
        batch_size, seq_len, num_heads, head_dim = key_states.shape

        if self.keys is None:
            dtype, device = key_states.dtype, key_states.device

            self.cache_position = 0
            self.max_seq_len = max(self.max_position_embeddings, seq_len)

            self.keys = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            self.values = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
        elif self.cache_position + seq_len >= self.max_seq_len:
            dtype, device = key_states.dtype, key_states.device

            self.max_seq_len = max(self.max_seq_len * 2, self.cache_position + seq_len)

            keys_new = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            values_new = infinicore.empty(
                [batch_size, self.max_seq_len, num_heads, head_dim],
                dtype=dtype,
                device=device,
            )
            keys_new.narrow(1, 0, self.cache_position).copy_(
                self.keys.narrow(1, 0, self.cache_position)
            )
            values_new.narrow(1, 0, self.cache_position).copy_(
                self.values.narrow(1, 0, self.cache_position)
            )

            self.keys, self.values = keys_new, values_new

    def update(
        self,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        # Lazy initialization
        self.lazy_initialization(key_states)

        seq_len = key_states.shape[1]
        index = self.cache_position

        # Update the cache
        self.keys.narrow(1, index, seq_len).copy_(key_states)
        self.values.narrow(1, index, seq_len).copy_(value_states)
        self.cache_position += seq_len

        return self.keys.narrow(1, 0, self.cache_position), self.values.narrow(
            1, 0, self.cache_position
        )


class Cache:
    """
    A `Cache` is mostly a list of `CacheLayerMixin` objects, one per model layer. It serves as a container for the Cache of each layer.

    Args:
        layers (`Optional`, *optional*): A list of pre-created `CacheLayerMixin`.
    """

    def __init__(
        self,
        layers: Optional[list[CacheLayerMixin]] = None,
    ):
        self.layers = layers if layers is not None else []

    def update(
        self,
        key_states: infinicore.Tensor,
        value_states: infinicore.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[infinicore.Tensor, infinicore.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`infinicore.Tensor`):
                The new key states to cache.
            value_states (`infinicore.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, *optional*):
                Additional arguments for the cache subclass.

        Return:
            A tuple containing the updated key and value states.
        """

        keys, values = self.layers[layer_idx].update(
            key_states, value_states, cache_kwargs
        )

        return keys.contiguous(), values.contiguous()


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of `CacheLayer`, one for each layer.

    Args:
        config (`PretrainedConfig`, *optional*):
            The config of the model for which this Cache will be used..

    """

    def __init__(
        self,
        config=None,
    ):
        max_position_embeddings = config.max_position_embeddings
        layers = []
        # If a config is passed, use it to infer the layer types and initialize accordingly
        if config is not None:
            config = config.get_text_config()
            layer_types = None
            if layer_types is None:
                layer_types = [
                    "full_attention" for _ in range(config.num_hidden_layers)
                ]

            for layer_type in layer_types:
                layers.append(DynamicLayer(max_position_embeddings))

        super().__init__(
            layers=layers,
        )
