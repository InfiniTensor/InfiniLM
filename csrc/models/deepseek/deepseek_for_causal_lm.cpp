#include "deepseek_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::deepseek {

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("deepseek" != model_type) {
        throw std::runtime_error(
            "infinilm::models::deepseek::create_deepseek_model_config: model_type is not deepseek");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size")
                                / model_config->get<size_t>("num_attention_heads");
    }

    config_json["num_experts"] = config_json.value("n_routed_experts", 0);
    config_json["mlp_bias"] = false;

    if (!config_json.contains("norm_topk_prob")) {
        config_json["norm_topk_prob"] = false;
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    if (!config_json.contains("attention_output_bias")) {
        config_json["attention_output_bias"] = config_json.value("attention_bias", false);
    }
    if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
        config_json["dtype"] = config_json["torch_dtype"];
    }

    return model_config;
}

} // namespace infinilm::models::deepseek

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek,
    infinilm::models::deepseek::DeepseekForCausalLM,
    infinilm::models::deepseek::create_deepseek_model_config);
} // namespace
