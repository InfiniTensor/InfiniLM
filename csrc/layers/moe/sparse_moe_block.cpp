#include "sparse_moe_block.hpp"

namespace infinilm::layers::moe {

SparseMoeBlock::SparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device,
                               size_t layer_id) {
    INFINICORE_NN_MODULE_INIT(gate, model_config, device);
    INFINICORE_NN_MODULE_INIT(experts, model_config, device);
    INFINICORE_NN_MODULE_INIT(fused_moe, model_config, device, layer_id);
}

infinicore::Tensor SparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    ASSERT(hidden_states->ndim() == 3);

    auto shape = hidden_states->shape();
    auto hidden_states_reshaped = hidden_states->view({shape[0] * shape[1], shape[2]});

    auto [routing_weights, selected_experts] = gate_->forward(hidden_states_reshaped);
    TopKOutput topk_output{
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

} // namespace infinilm::layers::moe
