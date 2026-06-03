#pragma once

#include "infinicore/nn/rope.hpp"
#include <memory>

namespace infinilm::config {
class ModelConfig; // Forward declaration
}

namespace infinilm::layers::rotary_embedding {

/**
 * @brief Public API to assemble and construct a complete RoPE module.
 *
 * @param model_config Model configuration.
 * @param device Device to create the cache on.
 * @param algo RoPE algorithm type (default: Algo::GPT_NEOX).
 */
std::shared_ptr<infinicore::nn::RoPE>
get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
         const infinicore::Device &device);

} // namespace infinilm::layers::rotary_embedding
