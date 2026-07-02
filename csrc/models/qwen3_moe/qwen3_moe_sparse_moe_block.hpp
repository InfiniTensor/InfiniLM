#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/moe/sparse_moe_block.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::qwen3_moe {

class Qwen3MoeSparseMoeBlock final : public infinilm::layers::moe::SparseMoeBlock {
public:
    // TextDecoderLayer constructs MLP blocks as (config, layer_idx, device),
    // while the reusable SparseMoeBlock takes (config, device, layer_idx).
    // Keep this adapter until those constructor signatures are unified.
    Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);
    Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);
};

} // namespace infinilm::models::qwen3_moe
