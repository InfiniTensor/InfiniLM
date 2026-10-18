#pragma once
#include "../../layers/rotary_embedding/rope.hpp"

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../weights/weights.hpp"
#include "runtime/device.hpp"
#include "runtime/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::ernie4_5_vl {

struct Ernie45MropeCache {
    std::shared_ptr<infinilm::runtime::Runtime> runtime_owner;
    void require_owner(infinilm::runtime::Runtime &runtime) const;
    std::vector<int> section{22, 22, 20};
    infinilm::runtime::Tensor cos_sin_h;
    infinilm::runtime::Tensor cos_sin_w;
    infinilm::runtime::Tensor cos_sin_t;
};

std::shared_ptr<const Ernie45MropeCache> build_ernie45_mrope_cache(infinilm::runtime::Runtime &runtime, std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                   const infinilm::runtime::Device &device);

class Ernie45Attention : public infinilm::weights::ModelWeights {
public:
    Ernie45Attention(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     std::shared_ptr<const Ernie45MropeCache> mrope_cache,
                     const engine::distributed::RankInfo &rank_info,
                     backends::AttentionBackend attention_backend);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      global_state::ForwardContext &forward_context) const;

private:
    infinilm::runtime::Tensor forward_static_(const infinilm::runtime::Tensor &positions,
                                              const infinilm::runtime::Tensor &hidden_states,
                                              global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor forward_paged_(const infinilm::runtime::Tensor &positions,
                                             const infinilm::runtime::Tensor &hidden_states,
                                             global_state::ForwardContext &forward_context) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t head_dim_{0};
    size_t rotary_dim_{0};
    size_t num_attention_heads_{0};
    size_t num_key_value_heads_{0};
    std::shared_ptr<const Ernie45MropeCache> mrope_cache_;
    infinilm::backends::AttentionBackend attention_backend_;

    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> q_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> k_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> v_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;

    infinilm::weights::Weight kv_cache_k_scale_;
    infinilm::weights::Weight kv_cache_v_scale_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
};

} // namespace infinilm::models::ernie4_5_vl
