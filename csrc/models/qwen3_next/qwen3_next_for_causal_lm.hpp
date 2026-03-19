#pragma once
#include "../../layers/common_modules.hpp"
#include "qwen3_next_decoderLayer.hpp"
#include "qwen3_next_sparse_moe_block.hpp"

namespace infinilm::models::qwen3_next {

/** @brief Qwen3 Next model architecture (without language modeling head) */
using Qwen3NextModel = infinilm::layers::TemplateModel<Qwen3NextDecoderLayer>;

/** @brief Qwen3 Next model for Causal Language Modeling */
using Qwen3NextForCausalLM = infinilm::layers::TemplateCausalLM<Qwen3NextModel>;

} // namespace infinilm::models::qwen3_next

namespace infinilm::models::qwen3_next {

static std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_next" != model_type) {
        throw std::runtime_error("create_qwen3_next_model_config: model_type is not qwen3_next");
    }

    nlohmann::json &config_json = model_config->get_config_json();
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

    if (!config_json.contains("qk_norm")) {
        config_json["qk_norm"] = true;
    }

    return model_config;
}

} // namespace infinilm::models::qwen3_next