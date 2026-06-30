#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/quantization/base_quantization.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

bool use_deepseek_v4_w8a8_linear(const std::shared_ptr<infinilm::config::ModelConfig> &model_config);

std::shared_ptr<infinilm::quantization::BaseQuantization> deepseek_v4_linear_quantization(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    bool use_quantization);

} // namespace infinilm::models::deepseek_v4
