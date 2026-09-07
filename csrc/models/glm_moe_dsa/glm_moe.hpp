#pragma once
#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/legacy/moe_mlp.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <infiniccl.h>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>
namespace infinilm::models::glm_moe_dsa {
using GlmDenseMLP = infinilm::layers::mlp::MLP;
using GlmExpertMLP = infinilm::layers::moe::legacy::MoeMLP;
class GlmTopKRouter final : public infinicore::nn::Module {
public:
    GlmTopKRouter(std::shared_ptr<infinilm::config::ModelConfig>, const infinicore::Device &);
    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &) const;
    void process_weights_after_loading() override;

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(e_score_correction_bias);
    infinicore::Tensor runtime_bias_;
    size_t num_experts_{0}, top_k_{0}, num_expert_group_{0}, topk_group_{0};
    bool renormalize_{false};
    float routed_scaling_factor_{1};
};
class GlmW4A8Experts final : public infinicore::nn::Module {
public:
    GlmW4A8Experts(std::shared_ptr<infinilm::config::ModelConfig>, const infinicore::Device &);
    infinicore::Tensor forward(const infinicore::Tensor &,
                               const infinicore::Tensor &,
                               const infinicore::Tensor &,
                               std::optional<infinicore::Tensor> = std::nullopt) const;

private:
    infinicore::Tensor w1_, s1_, w2_, s2_;
    size_t hidden_{0}, inter_{0}, nexpert_{0}, topk_{0}, tp_{1};
    infinicclComm_t comm_{nullptr};
};
class GlmMoE final : public infinicore::nn::Module {
public:
    GlmMoE(std::shared_ptr<infinilm::config::ModelConfig>, const infinicore::Device &);
    infinicore::Tensor forward(const infinicore::Tensor &) const;

private:
    INFINICORE_NN_MODULE(GlmTopKRouter, gate);
    INFINICORE_NN_MODULE(GlmW4A8Experts, experts);
    INFINICORE_NN_MODULE(GlmDenseMLP, shared_experts);
    bool shared_{false};
};
} // namespace infinilm::models::glm_moe_dsa
