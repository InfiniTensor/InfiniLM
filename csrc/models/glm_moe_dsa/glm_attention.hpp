#pragma once
#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
namespace infinilm::models::glm_moe_dsa {
class GlmAttention final : public infinicore::nn::Module {
public:
    GlmAttention(std::shared_ptr<infinilm::config::ModelConfig>, size_t, const infinicore::Device &);
    infinicore::Tensor forward(const infinicore::Tensor &, const infinicore::Tensor &) const;

private:
    size_t heads_{0}, qh_{0}, qn_{0}, qr_{0}, vh_{0}, ql_{0}, kvl_{0};
    float scale_{1};
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, q_a_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_a_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, q_b_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, kv_a_proj_with_mqa);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, kv_a_layernorm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, kv_b_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
    std::shared_ptr<infinicore::nn::RoPE> rope_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    infinicore::nn::Parameter ks_, vs_;
};
} // namespace infinilm::models::glm_moe_dsa
