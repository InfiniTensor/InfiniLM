#include "minicpm5_moe_for_causal_lm.hpp"

#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm5_moe {

std::shared_ptr<infinilm::config::ModelConfig>
create_minicpm5_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    // g9v3 is the HF model_type for G9 MoE; minicpm5_moe is the InfiniLM alias.
    if (model_type != "minicpm5_moe" && model_type != "g9v3") {
        throw std::runtime_error(
            "create_minicpm5_moe_model_config: model_type is not minicpm5_moe or g9v3");
    }

    auto &config_json = model_config->get_config_json();

    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size")
                                / model_config->get<size_t>("num_attention_heads");
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    if (!config_json.contains("attention_output_bias")) {
        config_json["attention_output_bias"] = config_json.value("attention_bias", false);
    }
    if (!config_json.contains("mlp_bias")) {
        config_json["mlp_bias"] = false;
    }
    if (!config_json.contains("rope_theta")) {
        config_json["rope_theta"] = 10000.0;
    }
    if (!config_json.contains("n_group")) {
        config_json["n_group"] = 1;
    }
    if (!config_json.contains("topk_group")) {
        config_json["topk_group"] = 1;
    }
    if (!config_json.contains("norm_topk_prob")) {
        config_json["norm_topk_prob"] = true;
    }
    if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
        config_json["dtype"] = config_json["torch_dtype"];
    }

    config_json["num_experts"] = config_json.value("n_routed_experts", 0);
    config_json["e_score_correction_bias"] = true;
    config_json["apply_routed_scaling_factor_on_output"] = false;
    // n_group/topk_group default to 1 → grouped selection is a no-op; use sigmoid
    // (same escape hatch as kimi_k25 until fused_gate is graph-safe on Denglin).
    config_json["moe_router_backend"] = "sigmoid";
    if (!config_json.contains("use_gated_attention")) {
        config_json["use_gated_attention"] = true;
    }

    // Llama / MiniCPM rotate_half = GPT_NEOX.
    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_NEOX);
    return model_config;
}

} // namespace infinilm::models::minicpm5_moe

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm5_moe,
    infinilm::models::minicpm5_moe::MiniCPM5MoEForCausalLM,
    infinilm::models::minicpm5_moe::create_minicpm5_moe_model_config);

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    g9v3,
    infinilm::models::minicpm5_moe::MiniCPM5MoEForCausalLM,
    infinilm::models::minicpm5_moe::create_minicpm5_moe_model_config);
} // namespace
