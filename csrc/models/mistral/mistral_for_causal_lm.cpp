#include "mistral_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include <limits>

namespace infinilm::models::mistral {

std::shared_ptr<infinilm::config::ModelConfig> create_mistral_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config) {
        throw std::invalid_argument("Mistral configuration is required");
    }
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("mistral" != model_type) {
        throw std::runtime_error(
            "infinilm::models::mistral::create_mistral_model_config: model_type is not mistral");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    // Only Mistral opts in. A missing/null declaration retains global attention.
    int64_t left = -1;
    if (config_json.contains("sliding_window") && !config_json.at("sliding_window").is_null()) {
        const auto &window = config_json.at("sliding_window");
        if (!window.is_number_integer() || window.is_boolean()
            || window.get<int64_t>() <= 0
            || window.get<int64_t>() > std::numeric_limits<int>::max() - 126LL) {
            throw std::invalid_argument("Mistral sliding_window must be null or a positive bounded integer");
        }
        left = window.get<int64_t>() - 1;
    }
    config_json["infinilm_window_size"] = {left, left < 0 ? -1 : 0};

    if (model_config->get<int64_t>("num_attention_heads") <= 0 || model_config->get<int64_t>("hidden_size") <= 0) {
        throw std::invalid_argument("Mistral head/hidden must be positive before division");
    }
    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
                        / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    infinilm::models::validate_dense_construction_metadata(model_config, "Mistral");
    return model_config;
}

} // namespace infinilm::models::mistral

namespace {

INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    mistral,
    infinilm::models::mistral::MistralForCausalLM,
    infinilm::models::mistral::create_mistral_model_config);

} // namespace
