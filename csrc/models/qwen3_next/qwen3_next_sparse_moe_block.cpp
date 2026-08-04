#include "qwen3_next_sparse_moe_block.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/mul.hpp"

#include <utility>

namespace infinilm::models::qwen3_next {

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : Qwen3NextSparseMoeBlock(model_config, 0, device) {
}

Qwen3NextSparseMoeBlock::Qwen3NextSparseMoeBlock(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : infinilm::layers::moe::SparseMoeBlock(model_config, device, layer_idx) {
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

infinicore::Tensor Qwen3NextSparseMoeBlock::forward(
    const infinicore::Tensor &hidden_states) const {
    auto routed_output = infinilm::layers::moe::SparseMoeBlock::forward(hidden_states);

    auto shared_output = shared_expert_->forward(hidden_states);
    auto shared_gate_input = hidden_states;
    auto shared_gate = infinicore::op::sigmoid(
        shared_expert_gate_->forward(shared_gate_input));
    shared_gate = shared_gate->as_strided(
        shared_output->shape(),
        {shared_gate->stride(0), shared_gate->stride(1), 0});
    shared_output = infinicore::op::mul(shared_output, shared_gate);

    return infinicore::op::add(routed_output, shared_output);
}

} // namespace infinilm::models::qwen3_next
