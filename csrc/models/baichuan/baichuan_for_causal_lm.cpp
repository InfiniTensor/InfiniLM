#include "baichuan_for_causal_lm.hpp"
#include "../llama/llama_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::baichuan {

std::shared_ptr<infinilm::config::ModelConfig> create_baichuan_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("baichuan" != model_type) {
        throw std::runtime_error(
            "infinilm::models::baichuan::create_baichuan_model_config: model_type is not baichuan");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("num_key_value_heads")) {
        config_json["num_key_value_heads"] = model_config->get<size_t>("num_attention_heads");
    }

    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size")
            / model_config->get<size_t>("num_attention_heads");
    }

    if (!config_json.contains("rope_theta")) {
        config_json["rope_theta"] = 10000.0;
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    return model_config;
}

} // namespace infinilm::models::baichuan

namespace {

#ifndef USE_CLASSIC_LLAMA

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    baichuan,
    infinilm::models::llama::LlamaForCausalLM,
    infinilm::models::baichuan::create_baichuan_model_config);

#endif

} // namespace
