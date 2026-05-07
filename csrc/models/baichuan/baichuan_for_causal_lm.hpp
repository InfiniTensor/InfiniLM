#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>

namespace infinilm::models::baichuan {

std::shared_ptr<infinilm::config::ModelConfig> create_baichuan_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::baichuan
