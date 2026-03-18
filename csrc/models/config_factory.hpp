#pragma once

#include "../config/model_config.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace infinilm {

/**
 * @brief Factory function type: take an already-loaded base ModelConfig,
 * return a (possibly modified) ModelConfig.
 */
using ConfigCreator = std::function<std::shared_ptr<infinilm::config::ModelConfig>(std::shared_ptr<infinilm::config::ModelConfig>)>;

class InfinilmConfigFactory {
public:
    static std::shared_ptr<infinilm::config::ModelConfig> createConfig(const std::string &model_path);

private:
    static std::map<std::string, ConfigCreator> &_modelConfigs();
};

} // namespace infinilm
