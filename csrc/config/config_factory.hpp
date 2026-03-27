#pragma once

#include "model_config.hpp"
#include <memory>
#include <string>

namespace infinilm::config {

class ConfigFactory {
public:
    static std::shared_ptr<infinilm::config::ModelConfig> createConfig(const std::string &model_path);
};

} // namespace infinilm::config
