#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../cache/kv_cache.hpp"
#include "../../../config/model_config.hpp"
#include "../../../engine/distributed/distributed.hpp"
#include "../../../models/infinilm_model.hpp"
#include "../../linear/fused_linear.hpp"
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
class FlashInferAttentionImpl {

public:
    FlashInferAttentionImpl(size_t num_heads,
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
    infinicore::Tensor forward(void *layer,
                               const infinicore::Tensor &query,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value,
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::InfinilmModel::Input &attn_metadata) const;

private:
    size_t num_heads_;
    size_t head_size_;
    float scale_;
    size_t num_kv_heads_;
    size_t layer_idx_;
    size_t head_dim_;
};
} // namespace infinilm::layers::attention::backends