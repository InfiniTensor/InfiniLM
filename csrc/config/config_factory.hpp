#pragma once

#include "model_config.hpp"
#include <memory>
#include <string>

namespace infinilm {

class InfinilmConfigFactory {
public:
    static std::shared_ptr<infinilm::config::ModelConfig> createConfig(const std::string &model_path);

};

} // namespace infinilm
