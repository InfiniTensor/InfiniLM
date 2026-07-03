#include "qwen3_moe_sparse_moe_block.hpp"

#include <string>

namespace infinilm::models::qwen3_moe {

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device)
    : Qwen3MoeSparseMoeBlock(model_config, 0, device) {
}

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device)
    : use_legacy_moe_(
        model_config->get_or<std::string>("model_type", "") == "qwen3_moe" && !model_config->get_or<bool>("skip_legacy_moe", false)) {
    if (use_legacy_moe_) {
        legacy_gate_ = this->register_module<Qwen3MoeTopKRouter>("gate", model_config, device);
        legacy_experts_ = this->register_module<Qwen3MoeExperts>("experts", model_config, device);
    } else {
        gate_ = this->register_module<infinilm::layers::moe::TopKRouter>("gate", model_config, device);
        experts_ = this->register_module<infinilm::layers::moe::FusedMoeExperts>("experts", model_config, device);
        fused_moe_ = this->register_module<infinilm::layers::moe::FusedMoE>("fused_moe", model_config, device, layer_idx);
    }
}

infinicore::Tensor Qwen3MoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    if (use_legacy_moe_) {
        auto [routing_weights, selected_experts] = legacy_gate_->forward(hidden_states_reshaped);
        auto final_hidden_states = legacy_experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);
        return final_hidden_states->view({shape[0], shape[1], shape[2]});
    }

    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
    infinilm::layers::moe::TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };

    auto final_hidden_states = fused_moe_->forward(
        hidden_states_reshaped,
        topk_output,
        experts_->moe_weights());

    return final_hidden_states->view({shape[0], shape[1], shape[2]});
}

} // namespace infinilm::models::qwen3_moe
