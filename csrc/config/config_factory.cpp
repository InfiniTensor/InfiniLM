#include "config_factory.hpp"
#include "../models/models_registry.hpp"
#include <stdexcept>

namespace infinilm::config {

std::shared_ptr<infinilm::config::ModelConfig> ConfigFactory::createConfig(const std::string &model_path) {
    auto model_config = std::make_shared<infinilm::config::ModelConfig>(model_path + "/config.json");
    if (nullptr == model_config) {
        throw std::runtime_error("infinilm::config::ConfigFactory::createConfig: model_config is not initialized");
    }

    const std::string model_type = model_config->get<std::string>("model_type");
    const auto &config_map = models::get_model_config_map();
    auto it = config_map.find(model_type);
    if (it != config_map.end()) {
        it->second(model_config);
    } else {
        std::vector<std::string> classic_models = {"llama", "qwen2", "minicpm", "fm9g", "fm9g7b"};
        const std::string &model_type = model_config->get<std::string>("model_type");
        if (std::find(classic_models.begin(), classic_models.end(), model_type) == classic_models.end()) {
            throw std::invalid_argument("infinilm::config::ConfigFactory::createConfig: Unsupported model config type: " + model_type);
        }
    }

    return model_config;
}

} // namespace infinilm::config
