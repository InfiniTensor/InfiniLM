#pragma once
#include "rope_scaling_configs.hpp"

#include "rope.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace infinilm::config {
class ModelConfig; // Forward declaration
}

namespace infinilm::layers::rotary_embedding {

std::shared_ptr<infinilm::layers::RoPE>
get_rope(infinilm::runtime::Runtime &runtime, const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
         size_t head_dim,
         size_t rotary_dim,
         size_t max_position_embeddings,
         double rope_theta,
         infinilm::layers::RoPE::Algo algo,
         const infinilm::runtime::DataType &dtype,
         const infinilm::runtime::Device &device,
         std::shared_ptr<infinilm::layers::RopeScalingConfig> scaling = nullptr,
         std::optional<std::vector<int>> mrope_section = std::nullopt,
         bool mrope_interleaved = false);

/**
 * @brief Public API to assemble and construct a complete RoPE module.
 *
 * @param model_config Model configuration.
 * @param device Device to create the cache on.
 * @todo Not Supported Yet: reading mrope params from config
 *
 */
std::shared_ptr<infinilm::layers::RoPE>
get_rope(infinilm::runtime::Runtime &runtime, const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
         const infinilm::runtime::Device &device);

} // namespace infinilm::layers::rotary_embedding
