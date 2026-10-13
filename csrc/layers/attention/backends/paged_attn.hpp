#pragma once

#include "../../../global_state/global_state.hpp"
#include "runtime/tensor.hpp"
#include <tuple>

namespace infinilm::layers::attention {
class AttentionLayer;
}
namespace infinilm::runtime { class Runtime; }

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
     * @return Attention output, shape `[1, num_tokens, num_heads * head_dim]`.
     */
    infinilm::runtime::Tensor forward(const AttentionLayer &layer,
                               const infinilm::runtime::Tensor &query,
                               const infinilm::runtime::Tensor &key,
                               const infinilm::runtime::Tensor &value,
                               infinilm::runtime::Tensor &kv_cache,
                               const infinilm::global_state::AttentionMetadata &attn_metadata,
                               infinilm::runtime::Runtime &runtime) const;

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> do_kv_cache_update(const AttentionLayer &layer,
                                                                          const infinilm::runtime::Tensor key,
                                                                          const infinilm::runtime::Tensor value,
                                                                          infinilm::runtime::Tensor &kv_cache,
                                                                          const infinilm::runtime::Tensor slot_mapping,
                                                                          infinilm::runtime::Runtime &runtime) const;

private:
    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_; // Note: head_dim equals to head_size
};
} // namespace infinilm::layers::attention::backends
