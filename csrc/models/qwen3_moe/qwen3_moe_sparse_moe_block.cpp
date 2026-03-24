#include "qwen3_moe_sparse_moe_block.hpp"
#include <cstdio>

#include "infinicore/io.hpp"
namespace infinilm::models::qwen3_moe {

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device) {

    const auto &dtype{model_config->get_dtype()};

    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t moe_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
    size_t shared_expert_intermediate_size = model_config->get_or<size_t>("shared_expert_intermediate_size", 0);
    size_t num_experts = model_config->get<size_t>("num_experts");

    INFINICORE_NN_MODULE_INIT(gate, hidden_size, num_experts, false, dtype, device);
    experts_.reserve(num_experts);
    for (size_t i = 0; i < num_experts; ++i) {
        experts_.push_back(this->register_module<Qwen3MoeMLP>("experts." + std::to_string(i), model_config, device));
    }

    if (shared_expert_intermediate_size > 0) {
        INFINICORE_NN_MODULE_INIT(shared_expert, model_config, device);
        INFINICORE_NN_MODULE_INIT(shared_expert_gate, hidden_size, 1, false, dtype, device);
    }
}

infinicore::Tensor Qwen3MoeSparseMoeBlock::forward(const infinicore::Tensor &hidden_states) const {
    spdlog::error("Qwen3MoeSparseMoeBlock: forward not implemented");
    return hidden_states;
}

} // namespace infinilm::models::qwen3_moe
