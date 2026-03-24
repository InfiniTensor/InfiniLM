#include "config_factory.hpp"
#include "models_registry.hpp"
#include <stdexcept>

namespace infinilm {

namespace {

std::map<std::string, models::ConfigCreator> &_modelConfigs() {
    static std::map<std::string, models::ConfigCreator> map = [] {
        std::map<std::string, models::ConfigCreator> m;
        models::register_model_configs(m);
        return m;
    }();
    return map;
}

} // namespace

std::shared_ptr<infinilm::config::ModelConfig> InfinilmConfigFactory::createConfig(const std::string &model_path) {
    auto model_config = std::make_shared<infinilm::config::ModelConfig>(model_path + "/config.json");
    if (nullptr == model_config) {
        throw std::runtime_error("InfinilmConfigFactory::createConfig: model_config is not initialized");
    }

    const std::string model_type = model_config->get<std::string>("model_type");
    auto &config_map = _modelConfigs();
    auto it = config_map.find(model_type);
    if (it != config_map.end()) {
        it->second(model_config);
    } else {
        throw std::invalid_argument("InfinilmConfigFactory::createConfig: Unsupported model config type: " + model_type);
    }

    return model_config;
}

} // namespace infinilm
