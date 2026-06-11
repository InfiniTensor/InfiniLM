#include "qwen2_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::qwen2 {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen2" != model_type) {
        throw std::runtime_error(
            "infinilm::models::qwen2::create_qwen2_model_config: model_type is not qwen2");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
                        / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }

    return model_config;
}

} // namespace infinilm::models::qwen2

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen2,
    infinilm::models::qwen2::Qwen2ForCausalLM,
    infinilm::models::qwen2::create_qwen2_model_config);

} // namespace
