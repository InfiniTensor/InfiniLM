#pragma once

#include "../qwen3/qwen3_for_causal_lm.hpp"

#include <variant>

namespace infinilm::models::qwen2 {

using Qwen2ForCausalLM = qwen3::Qwen3ForCausalLM;

static std::shared_ptr<infinilm::config::ModelConfig> create_qwen2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen2" != model_type) {
        throw std::runtime_error("create_qwen2_model_config: model_type is not qwen2");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("layer_types")) {
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; i++) {
            layer_types.push_back("full_attention");
        }
        config_json["layer_types"] = layer_types;
    }

    return model_config;
}

} // namespace infinilm::models::qwen2
