#pragma once

#include "../../layers/common_modules.hpp"
#include "qwen3_attention.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3 {

using Qwen3MLP = infinilm::layers::MLP;

using Qwen3Attention = infinilm::models::qwen3::Qwen3Attention;

using Qwen3DecoderLayer = infinilm::layers::TextDecoderLayer<Qwen3Attention, Qwen3MLP>;

using Qwen3Model = infinilm::layers::TextModel<Qwen3DecoderLayer>;

using Qwen3ForCausalLM = infinilm::layers::TextCausalLM<Qwen3Model>;

} // namespace infinilm::models::qwen3

namespace infinilm::models::qwen3 {

static std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {

    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen3" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3::create_qwen3_model_config: model_type is not qwen3");
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

    if (!config_json.contains("qk_norm")) {
        config_json["qk_norm"] = true;
    }

    return model_config;
}

} // namespace infinilm::models::qwen3
