#include "internlm3_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::internlm3 {

std::shared_ptr<infinilm::config::ModelConfig> create_internlm3_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("internlm3" != model_type) {
        throw std::runtime_error(
            "infinilm::models::internlm3::create_internlm3_model_config: model_type is not internlm3");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    return model_config;
}

} // namespace infinilm::models::internlm3

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    internlm3,
    infinilm::models::internlm3::InternLM3ForCausalLM,
    infinilm::models::internlm3::create_internlm3_model_config);
} // namespace
