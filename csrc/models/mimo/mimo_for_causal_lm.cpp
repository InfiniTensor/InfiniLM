#include "mimo_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::mimo {

std::shared_ptr<infinilm::config::ModelConfig> create_mimo_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("mimo" != model_type) {
        throw std::runtime_error(
            "infinilm::models::mimo::create_mimo_model_config: model_type is not mimo");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    // The released checkpoints carry head_dim and attention_bias, but a
    // converted config may rely on the architecture defaults of the same layer
    // stack: q/k/v projections carry biases and o_proj does not.
    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size")
                                / model_config->get<size_t>("num_attention_heads");
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = true;
    }

    return model_config;
}

} // namespace infinilm::models::mimo

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    mimo,
    infinilm::models::mimo::MiMoForCausalLM,
    infinilm::models::mimo::create_mimo_model_config);

} // namespace
