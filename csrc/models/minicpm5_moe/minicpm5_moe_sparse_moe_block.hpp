#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::minicpm5_moe {

std::shared_ptr<infinilm::config::ModelConfig>
make_minicpm5_mlp_config(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                         size_t intermediate_size);

/**
 * Sparse MoE + shared expert (DeepSeek-V3 / MiniCPM5 style).
 * Router: sigmoid + e_score_correction_bias; routed_scaling applied on routed output.
 */
class MiniCPM5MoESparseMoeBlock final : public infinicore::nn::Module {
public:
    MiniCPM5MoESparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                              size_t layer_idx,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::moe::TopKRouter, gate);
    INFINICORE_NN_MODULE(infinilm::layers::moe::FusedMoeExperts, experts);
    INFINICORE_NN_MODULE(infinilm::layers::moe::FusedMoE, fused_moe);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, shared_experts);
    bool has_shared_experts_{false};
    float routed_scaling_factor_{1.0f};
};

} // namespace infinilm::models::minicpm5_moe
