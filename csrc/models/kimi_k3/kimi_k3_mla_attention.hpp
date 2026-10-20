#pragma once
#include "../../weights/weights.hpp"

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/linear.hpp"

#include "../../layers/normalization/rmsnorm.hpp"
#include <runtime/tensor.hpp>

#include <memory>

namespace infinilm::models::kimi_k3 {

class KimiK3MLAAttention : public infinilm::weights::ModelWeights {
public:
    KimiK3MLAAttention(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &hidden_states,
                                      global_state::ForwardContext &forward_context) const;

private:
    size_t layer_idx_{0};
    size_t q_lora_rank_{0};
    size_t kv_lora_rank_{0};
    size_t qk_nope_head_dim_{0};
    size_t qk_rope_head_dim_{0};
    size_t q_head_dim_{0};
    size_t v_head_dim_{0};
    size_t local_num_heads_{0};
    float softmax_scale_{1.0f};

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> q_a_proj_;
    std::shared_ptr<infinilm::layers::RMSNorm> q_a_layernorm_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> q_b_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> kv_a_proj_with_mqa_;
    std::shared_ptr<infinilm::layers::RMSNorm> kv_a_layernorm_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> kv_b_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> g_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    infinilm::weights::Weight kv_cache_k_scale_;
    infinilm::weights::Weight kv_cache_v_scale_;
};

} // namespace infinilm::models::kimi_k3
