#include "hybrid_model_config.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::config {

void prepare_hybrid_model_config(
    const std::shared_ptr<ModelConfig> &model_config) {
    if (model_config == nullptr) {
        throw std::runtime_error(
            "prepare_hybrid_model_config: model_config is null");
    }

    auto &config_json = model_config->get_config_json();
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");

    if (!config_json.contains("layer_types")) {
        const size_t full_attention_interval = model_config->get<size_t>("full_attention_interval");
        if (full_attention_interval == 0) {
            throw std::runtime_error(
                "prepare_hybrid_model_config: full_attention_interval must be positive");
        }

        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            layer_types.push_back(
                (layer_idx + 1) % full_attention_interval == 0
                    ? "full_attention"
                    : "linear_attention");
        }
        config_json["layer_types"] = std::move(layer_types);
    }

    const auto &layer_types = config_json["layer_types"];
    if (!layer_types.is_array()
        || layer_types.size() != num_hidden_layers) {
        throw std::runtime_error(
            "prepare_hybrid_model_config: layer_types size must match num_hidden_layers");
    }
    for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
        if (!layer_types[layer_idx].is_string()) {
            throw std::runtime_error(
                "prepare_hybrid_model_config: layer_types entries must be strings");
        }
        const auto &layer_type = layer_types[layer_idx].get_ref<const std::string &>();
        if (layer_type != "full_attention"
            && layer_type != "linear_attention") {
            throw std::runtime_error(
                "prepare_hybrid_model_config: unsupported layer_type '"
                + layer_type + "' at layer " + std::to_string(layer_idx));
        }
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
}

} // namespace infinilm::config
