#pragma once

#include "../config/model_config.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace infinilm {

class InfinilmConfigFactory {
public:
    static std::shared_ptr<infinilm::config::ModelConfig> createConfig(const std::string &model_path);

};

} // namespace infinilm
