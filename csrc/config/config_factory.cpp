#include "config_factory.hpp"
#include "../models/models_registry.hpp"
#include <stdexcept>

namespace infinilm {

std::shared_ptr<infinilm::config::ModelConfig> InfinilmConfigFactory::createConfig(const std::string &model_path) {
    auto model_config = std::make_shared<infinilm::config::ModelConfig>(model_path + "/config.json");
    if (nullptr == model_config) {
        throw std::runtime_error("InfinilmConfigFactory::createConfig: model_config is not initialized");
    }

    const std::string model_type = model_config->get<std::string>("model_type");
    const auto &config_map = models::get_model_config_map();
    auto it = config_map.find(model_type);
    if (it != config_map.end()) {
        it->second(model_config);
    } else {
        model_config;
        // throw std::invalid_argument("InfinilmConfigFactory::createConfig: Unsupported model config type: " + model_type);
    }

    return model_config;
}

} // namespace infinilm
