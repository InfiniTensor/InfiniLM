#pragma once

#include "../../../engine/forward_context.hpp"
#include "infinicore/tensor.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace infinilm::layers::attention::backends {
/*
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py

class AttentionImpl(AttentionImplBase[T], Generic[T]):
    """Standard attention implementation with forward method."""
    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError
*/

/*
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py

class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        pass

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        pass
*/

class FlashAttentionImpl {

public:
    FlashAttentionImpl(size_t num_heads,
                       size_t head_size,
                       float scale,
                       size_t num_kv_heads,
                       size_t layer_idx);

    /*
    Forward pass with FlashAttention.

    Args:
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        kv_cache: PagedKVCache
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    */
    infinicore::Tensor forward(const void *layer,
                               const infinicore::Tensor &query,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value,
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::engine::AttentionMetadata &attn_metadata) const;

    std::tuple<infinicore::Tensor, infinicore::Tensor> do_kv_cache_update(const void *layer,
                                                                          const infinicore::Tensor key,
                                                                          const infinicore::Tensor value,
                                                                          std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                                                                          const infinicore::Tensor slot_mapping) const;

private:
    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_;
    size_t max_position_embeddings_;
};
} // namespace infinilm::layers::attention::backends
