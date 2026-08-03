#pragma once

#include "experts/fused_moe_experts.hpp"
#include "fused_moe.hpp"
#include "router/topk_router.hpp"

#include "../../config/model_config.hpp"
#include "../linear/linear.hpp"
#include "../mlp/mlp.hpp"
#include "infinicore/nn/module.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::layers::moe {

class SharedExpertSparseMoeBlock : public infinicore::nn::Module {
public:
    SharedExpertSparseMoeBlock(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);
    SharedExpertSparseMoeBlock(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(TopKRouter, gate);
    INFINICORE_NN_MODULE(FusedMoeExperts, experts);
    INFINICORE_NN_MODULE(FusedMoE, fused_moe);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, shared_expert);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, shared_expert_gate);
};

} // namespace infinilm::layers::moe
