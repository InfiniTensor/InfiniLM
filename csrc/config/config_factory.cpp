#include "config_factory.hpp"
#include "../models/models_registry.hpp"
#include <stdexcept>
#include <unordered_set>

namespace infinilm::config {

std::shared_ptr<infinilm::config::ModelConfig> ConfigFactory::createConfig(const std::string &config_str) {
    const nlohmann::json config_json = nlohmann::json::parse(config_str);
    auto model_config = std::make_shared<infinilm::config::ModelConfig>(config_json);

    const std::string model_type = model_config->get<std::string>("model_type");
    const auto &config_map = models::get_model_config_map();
    auto it = config_map.find(model_type);
    if (it == config_map.end()) {
        throw std::invalid_argument("infinilm::config::ConfigFactory::createConfig: Unsupported model config type: " + model_type);
    }

    static const std::unordered_set<std::string> kModernModelTypes{"qwen3"};
    if (kModernModelTypes.find(model_type) == kModernModelTypes.end()) {
        throw std::invalid_argument(
            "infinilm::config::ConfigFactory::createConfig: model type `" + model_type
            + "` is unavailable with the modern InfiniOps backend; supported model types: qwen3");
    }

    it->second(model_config);
    return model_config;
}

} // namespace infinilm::config
