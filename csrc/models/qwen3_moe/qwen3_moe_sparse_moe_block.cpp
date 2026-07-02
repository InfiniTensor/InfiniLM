#include "qwen3_moe_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_moe {

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device)
    : Qwen3MoeSparseMoeBlock(model_config, 0, device) {
}

Qwen3MoeSparseMoeBlock::Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device)
    : infinilm::layers::moe::SparseMoeBlock(model_config, device, layer_idx) {}

} // namespace infinilm::models::qwen3_moe
