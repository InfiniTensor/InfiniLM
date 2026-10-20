#include "qwen2_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::qwen2 {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config) throw std::invalid_argument("Qwen2 configuration is required");
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen2" != model_type) {
        throw std::runtime_error(
            "infinilm::models::qwen2::create_qwen2_model_config: model_type is not qwen2");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (model_config->get<int64_t>("num_attention_heads") <= 0 || model_config->get<int64_t>("hidden_size") <= 0)
        throw std::invalid_argument("Qwen2 head/hidden must be positive before division");
    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
                        / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }

    infinilm::models::validate_dense_construction_metadata(model_config, "Qwen2");
    return model_config;
}

} // namespace infinilm::models::qwen2

namespace {

INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    qwen2,
    infinilm::models::qwen2::Qwen2ForCausalLM,
    infinilm::models::qwen2::create_qwen2_model_config);

} // namespace
