#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../cache/kv_cache.hpp"
#include "../../../config/model_config.hpp"
#include "../../../engine/distributed/distributed.hpp"
#include "../../../models/infinilm_model.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace infinilm::layers::attention::backends {

class StaticAttentionImpl {

public:
    StaticAttentionImpl(size_t num_heads,
                        size_t head_size,
                        float scale,
                        size_t num_kv_heads,
                        size_t layer_idx);

    infinicore::Tensor forward(void *layer,
                               const infinicore::Tensor &q_reshaped, //  query
                               const infinicore::Tensor &k_permuted, // key
                               const infinicore::Tensor &v_permuted, // value
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::InfinilmModel::Input &attn_metadata) const;
    std::tuple<infinicore::Tensor, infinicore::Tensor> do_kv_cache_update(void *layer,
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
    size_t head_dim_;
};
} // namespace infinilm::layers::attention::backends