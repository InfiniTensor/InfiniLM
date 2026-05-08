#include "qwen3_moe_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_moe {

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
}

infinicore::Tensor Qwen3MoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    auto shape = hidden_states->shape(); // shape[ 1 11 2048 ]
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
    auto final_hidden_states = experts_->forward(hidden_states_reshaped, selected_experts, routing_weights);

    return final_hidden_states->view({shape[0], shape[1], shape[2]});
}

} // namespace infinilm::models::qwen3_moe
