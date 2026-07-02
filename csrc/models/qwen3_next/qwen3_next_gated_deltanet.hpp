#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3_next {
using Qwen3Next_Fake_RMSNormGated = infinicore::nn::RMSNorm;

class Qwen3NextGatedDeltaNet : public infinicore::nn::Module {
public:
    Qwen3NextGatedDeltaNet(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> in_proj_qkv_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> in_proj_z_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> in_proj_a_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> in_proj_b_;
    INFINICORE_NN_PARAMETER(conv1d_weight);
    INFINICORE_NN_PARAMETER(dt_bias);
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_MODULE(Qwen3Next_Fake_RMSNormGated, norm);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> out_proj_;

    size_t layer_idx_;
    size_t linear_num_value_heads_;
    size_t linear_num_key_heads_;
    size_t linear_key_head_dim_;
    size_t linear_value_head_dim_;
    size_t key_dim_;
    size_t value_dim_;
    size_t conv_dim_;
    size_t conv_state_len_;
};

} // namespace infinilm::models::qwen3_next
