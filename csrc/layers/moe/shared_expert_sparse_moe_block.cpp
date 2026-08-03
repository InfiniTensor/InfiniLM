#include "shared_expert_sparse_moe_block.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/mul.hpp"

#include <utility>

namespace infinilm::layers::moe {

SharedExpertSparseMoeBlock::SharedExpertSparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : SharedExpertSparseMoeBlock(model_config, 0, device) {
}

SharedExpertSparseMoeBlock::SharedExpertSparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    INFINICORE_NN_MODULE_INIT(fused_moe, model_config, device, layer_idx);

    auto shared_config_json = model_config->get_config_json();
    shared_config_json["intermediate_size"] = model_config->get<size_t>("shared_expert_intermediate_size");
    auto shared_config = std::make_shared<infinilm::config::ModelConfig>(
        std::move(shared_config_json));
    INFINICORE_NN_MODULE_INIT(shared_expert, shared_config, device);

    INFINICORE_NN_MODULE_INIT(
        shared_expert_gate,
        model_config->get<size_t>("hidden_size"),
        1,
        false,
        model_config->get_dtype(),
        device);
}

infinicore::Tensor SharedExpertSparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    const auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view(
        {shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
    TopKOutput topk_output{
        routing_weights,
        selected_experts,
        infinicore::Tensor(),
    };
    auto routed_output = fused_moe_->forward(
        hidden_states_reshaped,
        topk_output,
        experts_->moe_weights());

    auto shared_output = shared_expert_->forward(hidden_states);
    auto shared_gate_input = hidden_states;
    auto shared_gate = infinicore::op::sigmoid(
        shared_expert_gate_->forward(shared_gate_input));
    shared_gate = shared_gate->as_strided(
        shared_output->shape(),
        {shared_gate->stride(0), shared_gate->stride(1), 0});
    shared_output = infinicore::op::mul(shared_output, shared_gate);

    return infinicore::op::add(routed_output->view(shape), shared_output);
}

} // namespace infinilm::layers::moe
