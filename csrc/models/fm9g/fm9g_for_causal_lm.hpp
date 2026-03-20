#pragma once

#include "../../layers/common_modules.hpp"
#include <cstddef>
#include <variant>

namespace infinilm::models::fm9g {

/** @brief Type alias for fm9g MLP module */
using FM9GMLP = infinilm::layers::MLP;

using FM9GAttention = infinilm::layers::Attention;

/** @brief fm9g decoder layer type alias */
using FM9GDecoderLayer = infinilm::layers::TextDecoderLayer<FM9GAttention, FM9GMLP>;

/** @brief fm9g model architecture (without language modeling head) */
using FM9GModel = infinilm::layers::TextModel<FM9GDecoderLayer>;

/** @brief fm9g model for Causal Language Modeling */
using FM9GForCausalLM = infinilm::layers::TextCausalLM<FM9GModel>;

static std::shared_ptr<infinilm::config::ModelConfig> create_fm9g_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("llama" != model_type && "fm9g" != model_type) {
        throw std::runtime_error("create_fm9g_model_config: model_type is not llama or fm9g");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("head_dim")) {
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t num_attention_heads = model_config->get<size_t>("num_attention_heads");
        size_t head_dim = hidden_size / num_attention_heads;

        config_json["head_dim"] = head_dim;
    }

    if (!config_json.contains("mlp_bias")) {
        config_json["mlp_bias"] = false;
    }
    return model_config;
}
} // namespace infinilm::models::fm9g
