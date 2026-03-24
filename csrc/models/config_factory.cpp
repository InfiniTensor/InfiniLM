#include "config_factory.hpp"
#include "models.hpp"
#include <stdexcept>

namespace infinilm {

std::map<std::string, InfinilmConfigFactory::ConfigCreator> &InfinilmConfigFactory::_modelConfigs() {
    static std::map<std::string, ConfigCreator> _map;
    if (_map.empty()) {
        _map["llama"] = models::llama::create_llama_model_config;
        _map["fm9g"] = models::llama::create_llama_model_config;
        _map["qwen2"] = models::llama::create_llama_model_config;

        /////////
        // _map["qwen2"] = models::qwen2::create_qwen2_model_config;
        // _map["llama"] = models::fm9g::create_fm9g_model_config; // llama -> fm9g
        // _map["fm9g"] = models::fm9g::create_fm9g_model_config;
        _map["qwen3"] = models::qwen3::create_qwen3_model_config;
        _map["qwen3_moe"] = models::qwen3_moe::create_qwen3_moe_model_config;
        _map["qwen3_vl"] = models::qwen3_vl::create_qwen3_vl_model_config;
        _map["qwen3_next"] = models::qwen3_next::create_qwen3_next_model_config;
        _map["minicpm_sala"] = models::minicpm_sala::create_minicpm_sala_model_config;
    }
    return _map;
}

std::shared_ptr<infinilm::config::ModelConfig> InfinilmConfigFactory::createConfig(const std::string &model_path) {
    auto model_config = std::make_shared<infinilm::config::ModelConfig>(model_path + "/config.json");
    if (nullptr == model_config) {
        throw std::runtime_error("InfinilmConfigFactory::createConfig: model_config is not initialized");
    }

    const std::string model_type = model_config->get<std::string>("model_type");
    auto it = _modelConfigs().find(model_type);
    if (it != _modelConfigs().end()) {
        it->second(model_config);
    } else {
        throw std::invalid_argument("InfinilmConfigFactory::createConfig: Unsupported model config type: " + model_type);
    }

    return model_config;
}

} // namespace infinilm
