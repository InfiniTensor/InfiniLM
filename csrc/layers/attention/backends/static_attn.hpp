#pragma once

#include "../../../engine/forward_context.hpp"
#include "infinicore/tensor.hpp"
#include <tuple>

namespace infinilm::layers::attention {
class AttentionLayer;
}

namespace infinilm::layers::attention::backends {

class StaticAttentionImpl {
public:
    StaticAttentionImpl(size_t num_heads,
                        size_t head_size,
                        float scale,
                        size_t num_kv_heads,
                        size_t layer_idx);

    infinicore::Tensor forward(const AttentionLayer &layer,
                               const infinicore::Tensor &q_reshaped, // query
                               const infinicore::Tensor &k_permuted, // key
                               const infinicore::Tensor &v_permuted, // value
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::engine::AttentionMetadata &attn_metadata) const;

    std::tuple<infinicore::Tensor, infinicore::Tensor> do_kv_cache_update(const AttentionLayer &layer,
                                                                          const infinicore::Tensor key,
                                                                          const infinicore::Tensor value,
                                                                          std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                                                                          const infinicore::Tensor past_sequence_lengths) const;

private:
    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_; // Note: head_dim equals to head_size
};
} // namespace infinilm::layers::attention::backends
