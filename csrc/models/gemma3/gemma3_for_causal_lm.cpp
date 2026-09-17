#include "gemma3_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::gemma3 {

std::shared_ptr<infinilm::config::ModelConfig> create_gemma3_text_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("gemma3_text" != model_type) {
        throw std::runtime_error(
            "infinilm::models::gemma3::create_gemma3_text_model_config: model_type is not gemma3_text");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    // Linear RoPE scaling (gemma-3-4b/12b/27b text configs use factor 8 on the
    // global layers to reach 128k context) is not implemented. Refuse loudly
    // instead of silently building global layers with unscaled RoPE, which
    // corrupts long-context outputs. The text-only gemma-3-1b has no
    // rope_scaling and is unaffected.
    if (config_json.contains("rope_scaling") && !config_json["rope_scaling"].is_null()) {
        throw std::runtime_error(
            "infinilm::models::gemma3::create_gemma3_text_model_config: rope_scaling is not supported "
            "(gemma-3-4b/12b/27b use linear scaling on global attention layers); refusing to load a "
            "checkpoint whose global-layer RoPE cannot be reproduced");
    }

    // Gemma-3 has a dedicated head_dim that is NOT hidden_size / num_attention_heads.
    if (!config_json.contains("head_dim")) {
        if (config_json.contains("query_pre_attn_scalar")) {
            config_json["head_dim"] = model_config->get<size_t>("query_pre_attn_scalar");
        } else {
            throw std::runtime_error(
                "infinilm::models::gemma3::create_gemma3_text_model_config: config lacks head_dim and query_pre_attn_scalar");
        }
    }

    // The generic attention module defaults attention_bias to true; Gemma-3 has none.
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    // Only the tanh-approximated GELU is implemented (matches the checkpoints).
    const std::string activation = config_json.value("hidden_activation", "gelu_pytorch_tanh");
    if (activation != "gelu_pytorch_tanh") {
        throw std::runtime_error(
            "infinilm::models::gemma3::create_gemma3_text_model_config: unsupported hidden_activation: " + activation);
    }

    // Generate the per-layer attention types exactly like HF Gemma3TextConfig:
    // every `sliding_window_pattern`-th layer (default 6 -> 5:1) is full attention.
    if (!config_json.contains("layer_types")) {
        size_t num_layers = model_config->get<size_t>("num_hidden_layers");
        size_t pattern = config_json.value("sliding_window_pattern", 6);
        nlohmann::json layer_types = nlohmann::json::array();
        for (size_t i = 0; i < num_layers; ++i) {
            layer_types.push_back(((i + 1) % pattern != 0) ? "sliding_attention" : "full_attention");
        }
        config_json["layer_types"] = layer_types;
    }

    return model_config;
}

} // namespace infinilm::models::gemma3

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    gemma3_text,
    infinilm::models::gemma3::Gemma3ForCausalLM,
    infinilm::models::gemma3::create_gemma3_text_model_config);

} // namespace
