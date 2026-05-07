#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::glm4 {

std::shared_ptr<infinilm::config::ModelConfig> create_glm4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::glm4
