#include "minicpm5_moe_for_causal_lm.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm5_moe {

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if (model_type != "minicpm5_moe") {
        throw std::runtime_error("create_minicpm5_moe_model_config: model_type is not minicpm5_moe");
    }

    auto &j = model_config->get_config_json();
    if (!j.contains("rope_theta")) {
        j["rope_theta"] = 10000.0;
    }
    if (!j.contains("num_experts") && j.contains("n_routed_experts")) {
        j["num_experts"] = j["n_routed_experts"];
    }
    if (!j.contains("n_group")) {
        j["n_group"] = 1;
    }
    if (!j.contains("topk_group")) {
        j["topk_group"] = 1;
    }
    if (!j.contains("norm_topk_prob")) {
        j["norm_topk_prob"] = true;
    }
    if (!j.contains("attention_bias")) {
        j["attention_bias"] = false;
    }
    if (!j.contains("mlp_bias")) {
        j["mlp_bias"] = false;
    }
    // LongRoPE scales are present in HF config; InfiniLM RoPE factory tolerates rope_scaling dict.
    return model_config;
}

} // namespace infinilm::models::minicpm5_moe

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm5_moe,
    infinilm::models::minicpm5_moe::MiniCPM5MoeForCausalLM,
    infinilm::models::minicpm5_moe::create_minicpm5_moe_model_config);
} // namespace
