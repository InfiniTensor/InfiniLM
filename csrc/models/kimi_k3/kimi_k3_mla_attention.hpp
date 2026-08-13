#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/linear.hpp"

#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::kimi_k3 {

class KimiK3MLAAttention : public infinicore::nn::Module {
public:
    KimiK3MLAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

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
    backends::AttentionBackend attention_backend_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, q_a_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_a_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, q_b_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, kv_a_proj_with_mqa);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, kv_a_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, kv_b_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, g_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    infinicore::nn::Parameter kv_cache_k_scale_;
    infinicore::nn::Parameter kv_cache_v_scale_;
};

} // namespace infinilm::models::kimi_k3
