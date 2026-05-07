#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::chatglm {

std::shared_ptr<infinilm::config::ModelConfig> create_chatglm_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::chatglm
