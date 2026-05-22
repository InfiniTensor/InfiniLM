#include "glm4_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include <stdexcept>

namespace infinilm::models::glm4 {

std::shared_ptr<infinilm::config::ModelConfig> create_glm4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("glm4" != model_type) {
        throw std::runtime_error(
            "infinilm::models::glm4::create_glm4_model_config: model_type is not glm4");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    return model_config;
}

} // namespace infinilm::models::glm4

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    glm4,
    infinilm::models::glm4::Glm4ForCausalLM,
    infinilm::models::glm4::create_glm4_model_config);
} // namespace
