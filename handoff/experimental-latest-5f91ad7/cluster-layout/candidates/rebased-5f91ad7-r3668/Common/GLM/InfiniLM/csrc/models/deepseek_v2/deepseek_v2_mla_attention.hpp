#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../layers/rotary_embedding/rope.hpp"
#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v2 {

class DeepseekV2MLAAttention : public infinilm::weights::ModelWeights {
public:
    DeepseekV2MLAAttention(infinilm::runtime::Runtime &runtime,
                           std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const engine::distributed::RankInfo &rank_info,
                           backends::AttentionBackend attention_backend);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      infinilm::global_state::ForwardContext &forward_context) const;

private:
    infinilm::runtime::Tensor forward_static_(const infinilm::runtime::Tensor &positions,
                                              const infinilm::runtime::Tensor &hidden_states,
                                              infinilm::global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor forward_paged_(const infinilm::runtime::Tensor &positions,
                                             const infinilm::runtime::Tensor &hidden_states,
                                             infinilm::global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor position_ids_for_rope_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &position_ids) const;
    infinilm::runtime::Tensor kv_b_weight_3d_() const;
    infinilm::runtime::Tensor project_q_nope_to_latent_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &q_nope) const;
    infinilm::runtime::Tensor project_latent_to_value_(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &attn_output,
                                                       size_t batch_size,
                                                       size_t seq_len) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t num_attention_heads_{0};
    size_t qk_nope_head_dim_{0};
    size_t qk_rope_head_dim_{0};
    size_t q_head_dim_{0};
    size_t v_head_dim_{0};
    size_t kv_lora_rank_{0};
    size_t mla_head_dim_{0};
    float softmax_scale_{1.0f};
    infinilm::backends::AttentionBackend attention_backend_;

    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> q_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> kv_a_proj_with_mqa_;
    std::shared_ptr<infinilm::layers::RMSNorm> kv_a_layernorm_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> kv_b_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;

    std::shared_ptr<infinilm::layers::RoPE> rotary_emb_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> latent_attn_;
    infinilm::weights::Weight kv_cache_k_scale_;
    infinilm::weights::Weight kv_cache_v_scale_;
};

} // namespace infinilm::models::deepseek_v2
