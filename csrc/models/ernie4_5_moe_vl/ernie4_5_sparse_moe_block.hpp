#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"
#include "../../layers/moe/sparse_moe_block.hpp"

#include <memory>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5SparseMoeBlock final : public infinilm::layers::moe::SparseMoeBlock {
public:
    Ernie4_5SparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device,
                           size_t layer_id = 0);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    infinicore::Tensor forward_routed_(const infinicore::Tensor &hidden_states,
                                       const infinilm::layers::moe::TopKRouter &gate,
                                       const infinilm::layers::moe::FusedMoeExperts &experts,
                                       const infinilm::layers::moe::FusedMoE &fused_moe) const;

    infinicore::Tensor forward_text_(const infinicore::Tensor &hidden_states) const;
    infinicore::Tensor forward_vision_(const infinicore::Tensor &hidden_states) const;

    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, shared_experts);
    INFINICORE_NN_MODULE(infinilm::layers::moe::TopKRouter, vision_gate);
    INFINICORE_NN_MODULE(infinilm::layers::moe::FusedMoeExperts, vision_experts);
    INFINICORE_NN_MODULE(infinilm::layers::moe::FusedMoE, vision_fused_moe);

    bool has_shared_experts_{false};
    bool has_vision_experts_{false};
};

} // namespace infinilm::models::ernie4_5_moe_vl
