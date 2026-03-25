#pragma once

#include "../../../engine/forward_context.hpp"
#include "infinicore/tensor.hpp"
#include <cstddef>
#include <tuple>

namespace infinilm::layers::attention {
class AttentionLayer;
}

namespace infinilm::layers::attention::backends {

class PagedAttentionImpl {

public:
    PagedAttentionImpl(size_t num_heads,
                       size_t head_size,
                       float scale,
                       size_t num_kv_heads,
                       size_t layer_idx);

    /**
     * @brief Forward pass with PagedAttention.
     *
     * @param layer: The `AttentionLayer` instance.
     * @param query: Query tensor, shape `[num_tokens, num_heads, head_dim]`.
     * @param key: Key tensor, shape `[num_tokens, num_kv_heads, head_dim]`.
     * @param value: Value tensor, shape `[num_tokens, num_kv_heads, head_dim]`.
     * @param kv_cache: `(k_cache, v_cache)` paged KV tensors for this layer.
     * @param attn_metadata: Attention metadata.
     * @return Attention output, shape `[num_tokens, num_heads * head_dim]`.
     */
    infinicore::Tensor forward(const AttentionLayer &layer,
                               const infinicore::Tensor &query,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value,
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::engine::AttentionMetadata &attn_metadata) const;

    std::tuple<infinicore::Tensor, infinicore::Tensor> do_kv_cache_update(const AttentionLayer &layer,
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
    size_t head_dim_; // Note: head_dim equals to head_size
};
} // namespace infinilm::layers::attention::backends
