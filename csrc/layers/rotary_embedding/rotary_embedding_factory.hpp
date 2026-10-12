#pragma once

#include "rope.hpp"
#include "rope_scaling_configs.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace infinilm::config {
class ModelConfig; // Forward declaration
}

namespace infinilm::layers::rotary_embedding {

/**
 * @brief Function pointer type for creating specific RopeScalingConfig instances.
 * Implementations should extract parameters from ModelConfig and construct the corresponding Config object.
 */
using ScalingCreator = std::function<std::shared_ptr<infinilm::layers::RopeScalingConfig>(
    const std::shared_ptr<infinilm::config::ModelConfig> &)>;

/**
 * @brief Get the singleton registry mapping scaling type strings to their creator functions.
 */
std::unordered_map<std::string, ScalingCreator> &get_scaling_registry();

/**
 * @brief Factory method to create a RopeScalingConfig based on the ModelConfig.
 * Routes the "rope_scaling_type" string to the corresponding registered creator.
 */
std::shared_ptr<infinilm::layers::RopeScalingConfig>
make_scaling_config(const std::shared_ptr<infinilm::config::ModelConfig> &model_config);

} // namespace infinilm::layers::rotary_embedding
