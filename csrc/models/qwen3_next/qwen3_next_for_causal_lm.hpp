#pragma once
#include "../../layers/common_modules.hpp"
#include <memory>
#include <variant>

namespace infinilm::models::qwen3_next {

using Qwen3NextRMSNormGated = infinicore::nn::RMSNorm;

using StaticAttn = infinilm::models::layers::StaticAttention;
using PagedAttn = infinilm::models::layers::PagedAttention;
using FlashAttn = infinilm::models::layers::FlashAttention;
using Qwen3NextAttention = std::variant<std::shared_ptr<StaticAttn>, std::shared_ptr<PagedAttn>, std::shared_ptr<FlashAttn>>;

using Qwen3NextSparseMoeBlock = infinilm::models::qwen3_moe::Qwen3MoeSparseMoeBlock;

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

    return model_config;
}

} // namespace infinilm::models::qwen3_next