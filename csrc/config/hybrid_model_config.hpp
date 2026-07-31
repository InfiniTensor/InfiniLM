#pragma once

#include "model_config.hpp"

#include <memory>

namespace infinilm::config {

void prepare_hybrid_model_config(
    const std::shared_ptr<ModelConfig> &model_config);

} // namespace infinilm::config
