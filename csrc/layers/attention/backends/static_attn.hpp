#pragma once

#include "../../../global_state/global_state.hpp"
#include "../../../layers/quantization/kv_quant.hpp"
#include "infinicore/nn/module.hpp"
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
                               infinicore::Tensor &q_reshaped, // query
                               infinicore::Tensor &k_permuted, // key
                               infinicore::Tensor &v_permuted, // value
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::global_state::AttentionMetadata &attn_metadata) const;

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

    infinicore::quantization::KVQuantAlgo kv_quant_scheme_;
};
} // namespace infinilm::layers::attention::backends
