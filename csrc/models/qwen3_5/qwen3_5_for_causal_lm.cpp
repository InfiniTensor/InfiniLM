#include "qwen3_5_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config") && config_json["text_config"].is_object()) {
        const nlohmann::json &text_config_json = config_json["text_config"];
        for (auto it = text_config_json.begin(); it != text_config_json.end(); ++it) {
            if (!config_json.contains(it.key())) {
                config_json[it.key()] = it.value();
            }
        }
        if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
            config_json["dtype"] = config_json["torch_dtype"];
        }
    }
    if (!config_json.contains("position_id_axes")) {
        size_t position_id_axes = 1;
        if (config_json.contains("rope_parameters")
            && config_json["rope_parameters"].is_object()) {
            const auto &rope_parameters = config_json["rope_parameters"];
            if (rope_parameters.contains("mrope_section")
                && rope_parameters["mrope_section"].is_array()
                && !rope_parameters["mrope_section"].empty()) {
                position_id_axes = rope_parameters["mrope_section"].size();
            }
        }
        config_json["position_id_axes"] = position_id_axes;
    }
    if (!config_json.contains("rope_theta") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("rope_theta")) {
        // Normalize the nested HuggingFace field for the Qwen3.5 attention module.
        config_json["rope_theta"] = config_json["rope_parameters"]["rope_theta"];
    }
    if (!config_json.contains("partial_rotary_factor") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("partial_rotary_factor")) {
        config_json["partial_rotary_factor"] = config_json["rope_parameters"]["partial_rotary_factor"];
    }
    if (!config_json.contains("layer_types")) {
        size_t full_attention_interval = model_config->get<size_t>("full_attention_interval");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; i++) {
            layer_types.push_back(bool((i + 1) % full_attention_interval) ? "linear_attention" : "full_attention");
        }
        config_json["layer_types"] = layer_types;
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    return model_config;
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_5" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_5::create_qwen3_5_model_config: model_type is not qwen3_5");
    }
    return prepare_qwen3_5_model_config(model_config);
}

} // namespace infinilm::models::qwen3_5

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_5,
    infinilm::models::qwen3_5::Qwen35ForCausalLM,
    infinilm::models::qwen3_5::create_qwen3_5_model_config);
} // namespace
