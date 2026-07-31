#include "qwen3_5_moe_sparse_moe_block.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/mul.hpp"

namespace infinilm::models::qwen3_5_moe {

Qwen35MoeSparseMoeBlock::Qwen35MoeSparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    use_legacy_moe_ = model_config->get_or<bool>("use_legacy_moe", false);
    if (use_legacy_moe_) {
        legacy_gate_ = this->register_module<infinilm::models::qwen3_moe::Qwen3MoeTopKRouter>(
            "gate", model_config, device);
        legacy_experts_ = this->register_module<infinilm::models::qwen3_moe::Qwen3MoeExperts>(
            "experts", model_config, device);
    } else {
        gate_ = this->register_module<infinilm::layers::moe::TopKRouter>("gate", model_config, device);
        experts_ = this->register_module<infinilm::layers::moe::FusedMoeExperts>("experts", model_config, device);
        fused_moe_ = this->register_module<infinilm::layers::moe::FusedMoE>("fused_moe", model_config, device, layer_idx);
    }

    auto shared_config_json = model_config->get_config_json();
    shared_config_json["intermediate_size"] = model_config->get<size_t>("shared_expert_intermediate_size");
    auto shared_config = std::make_shared<infinilm::config::ModelConfig>(shared_config_json);
    INFINICORE_NN_MODULE_INIT(shared_expert, shared_config, device);

    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    INFINICORE_NN_MODULE_INIT(shared_expert_gate, hidden_size, 1, false, dtype, device);
}

infinicore::Tensor Qwen35MoeSparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    const auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    infinicore::Tensor expert_output;
    if (use_legacy_moe_) {
        auto [routing_weights, selected_experts] = legacy_gate_->forward(hidden_states_reshaped);
        expert_output = legacy_experts_->forward(
            hidden_states_reshaped,
            selected_experts,
            routing_weights);
    } else {
        auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
        infinilm::layers::moe::TopKOutput topk_output{
            routing_weights,
            selected_experts,
            infinicore::Tensor(),
        };
        expert_output = fused_moe_->forward(
            hidden_states_reshaped,
            topk_output,
            experts_->moe_weights());
    }

    auto shared_output = shared_expert_->forward(hidden_states);
    auto shared_gate_input = hidden_states;
    auto shared_gate = infinicore::op::sigmoid(shared_expert_gate_->forward(shared_gate_input));
    const auto &shared_shape = shared_output->shape();
    auto shared_gate_broadcast = infinicore::op::broadcast_to(
        shared_gate,
        {
            static_cast<int64_t>(shared_shape[0]),
            static_cast<int64_t>(shared_shape[1]),
            static_cast<int64_t>(shared_shape[2]),
        });
    shared_output = infinicore::op::mul(shared_output, shared_gate_broadcast);

    return infinicore::op::add(expert_output->view(shape), shared_output);
}

} // namespace infinilm::models::qwen3_5_moe
