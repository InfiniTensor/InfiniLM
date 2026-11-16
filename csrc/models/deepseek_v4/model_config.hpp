#pragma once

#include "config/model_config.hpp"

namespace infinilm::models::deepseek_v4 {

// Normalize checkpoint fields needed by the native V4 model without changing
// the user's config.json or interpreting packed FP4 weights as INT8.
std::shared_ptr<config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<config::ModelConfig> source);

} // namespace infinilm::models::deepseek_v4
