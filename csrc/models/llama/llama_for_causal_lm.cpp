#include "llama_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::llama {

std::shared_ptr<infinilm::config::ModelConfig> create_llama_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("llama" != model_type) {
        throw std::runtime_error(
            "infinilm::models::llama::create_llama_model_config: model_type is not llama");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size")
                                / model_config->get<size_t>("num_attention_heads");
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    return model_config;
}

} // namespace infinilm::models::llama

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    llama,
    infinilm::models::llama::LlamaForCausalLM,
    infinilm::models::llama::create_llama_model_config);

} // namespace
