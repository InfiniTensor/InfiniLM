#pragma once

#include "../config/model_config.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace infinilm {

class InfinilmConfigFactory {
    using ConfigCreator = std::function<std::shared_ptr<infinilm::config::ModelConfig>(std::shared_ptr<infinilm::config::ModelConfig>)>;

public:
    static std::shared_ptr<infinilm::config::ModelConfig> createConfig(const std::string &model_path);

private:
    static std::map<std::string, ConfigCreator> &_modelConfigs();
};

} // namespace infinilm
