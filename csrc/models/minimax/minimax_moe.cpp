#include "minimax_moe.hpp"

#include <infinicore/ops/add.hpp>

namespace infinilm::models::minimax {

MiniMaxMoeBlock::MiniMaxMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 size_t layer_idx,
                                 const infinicore::Device &device) {
    gate_ = this->register_module<infinilm::layers::moe::TopKRouter>("gate", model_config, device);
    experts_ = this->register_module<infinilm::layers::moe::FusedMoeExperts>("experts", model_config, device);
    fused_moe_ = this->register_module<infinilm::layers::moe::FusedMoE>("fused_moe", model_config, device, layer_idx);
}

infinicore::Tensor MiniMaxMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    auto shape = hidden_states->shape();
    auto hidden_flat = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(hidden_flat);
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };
    auto routed_states = fused_moe_->forward(hidden_flat, topk_output, experts_->moe_weights());

    return routed_states->as_strided(
        {shape[0], shape[1], shape[2]},
        {static_cast<infinicore::Stride>(shape[1] * shape[2]),
         static_cast<infinicore::Stride>(shape[2]),
         1});
}

} // namespace infinilm::models::minimax
