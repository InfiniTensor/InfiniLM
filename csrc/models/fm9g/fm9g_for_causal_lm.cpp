#include "fm9g_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::fm9g {

std::shared_ptr<infinilm::config::ModelConfig> create_fm9g_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }
    return model_config;
}

} // namespace infinilm::models::fm9g

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    fm9g,
    infinilm::models::fm9g::FM9GForCausalLM,
    infinilm::models::fm9g::create_fm9g_model_config);

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    fm9g7b,
    infinilm::models::fm9g::FM9GForCausalLM,
    infinilm::models::fm9g::create_fm9g_model_config);

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm,
    infinilm::models::fm9g::FM9GForCausalLM,
    infinilm::models::fm9g::create_fm9g_model_config);

} // namespace
