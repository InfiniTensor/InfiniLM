#pragma once

#include "experts/fused_moe_experts.hpp"
#include "fused_moe.hpp"
#include "router/topk_router.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/module.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::layers::moe {

class SparseMoeBlock : public infinicore::nn::Module {
public:
    SparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device,
                   size_t layer_id = 0);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(TopKRouter, gate);
    INFINICORE_NN_MODULE(FusedMoeExperts, experts);
    INFINICORE_NN_MODULE(FusedMoE, fused_moe);
};

} // namespace infinilm::layers::moe
