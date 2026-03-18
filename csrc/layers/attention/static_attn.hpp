#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../models/infinilm_model.hpp"
#include "attn_base.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace infinilm::models::layers::attention {
/**
 * @brief Multi-head self-attention module (static KV cache)
 *
 * Implements the attention mechanism with:
 * - Query, Key, Value projections
 * - Output projection
 * - Rotary Position Embeddings (RoPE) applied to Q and K
 * - Support for Grouped Query Attention (GQA)
 * - StaticKVCache only (no PagedKVCache)
 */
class StaticAttention : public AttentionBase {
public:
    StaticAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device,
                    size_t layer_idx,
                    engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinilm::InfinilmModel::Input &input,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const;
};

} // namespace infinilm::models::layers::attention
