#pragma once

#include "../../../global_state/global_state.hpp"
#include "infinicore/tensor.hpp"
#include <tuple>

namespace infinilm::layers::attention {
class AttentionLayer;
}

namespace infinilm::layers::attention::backends {

class HybridAttentionImpl {
public:
    HybridAttentionImpl(size_t num_heads,
                        size_t head_size,
                        float scale,
                        size_t num_kv_heads,
                        size_t layer_idx);

    infinicore::Tensor forward(const AttentionLayer &layer,
                               const infinicore::Tensor &query,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value,
                               infinicore::Tensor &kv_cache,
                               const infinilm::global_state::AttentionMetadata &attn_metadata) const;

    std::tuple<infinicore::Tensor, infinicore::Tensor> do_kv_cache_update(const infinicore::Tensor key,
                                                                          const infinicore::Tensor value,
                                                                          infinicore::Tensor &kv_cache,
                                                                          const infinicore::Tensor slot_mapping) const;

private:
    size_t num_heads_;
    float scale_;
    size_t head_dim_;
    size_t max_position_embeddings_;
};

} // namespace infinilm::layers::attention::backends
