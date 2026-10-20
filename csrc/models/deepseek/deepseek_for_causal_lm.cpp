#include "deepseek_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace infinilm::models::deepseek {

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config) {
        throw std::invalid_argument("Deepseek requires configuration");
    }
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("deepseek" != model_type) {
        throw std::runtime_error(
            "infinilm::models::deepseek::create_deepseek_model_config: model_type is not deepseek");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    for (const char *key : {"hidden_size", "vocab_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "max_position_embeddings"}) {
        if (model_config->get<int64_t>(key) <= 0) {
            throw std::invalid_argument("Deepseek requires positive construction dimensions");
        }
    }
    if (model_config->get<int64_t>("num_hidden_layers") < 0) {
        throw std::invalid_argument("Deepseek requires nonnegative layer count");
    }
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t heads = model_config->get<size_t>("num_attention_heads");
    if (hidden % heads) {
        throw std::invalid_argument("Deepseek hidden_size must be divisible by num_attention_heads");
    }
    const size_t head_dim = hidden / heads;
    if (config_json.contains("head_dim") && !config_json.at("head_dim").is_null()
        && (!config_json.at("head_dim").is_number_integer()
            || config_json.at("head_dim").get<size_t>() != head_dim)) {
        throw std::invalid_argument("Deepseek head_dim must equal hidden_size / num_attention_heads");
    }
    config_json["head_dim"] = head_dim;

    // Official DeepseekConfig uses nullable counts for the dense model.
    for (const char *key : {"n_routed_experts", "n_shared_experts"}) {
        if (!config_json.contains(key) || config_json.at(key).is_null()) {
            config_json[key] = 0;
        }
        if (!config_json.at(key).is_number_integer() || config_json.at(key).get<int64_t>() < 0) {
            throw std::invalid_argument("Deepseek expert counts must be nonnegative integers or null");
        }
    }
    if (config_json.value("hidden_act", std::string("silu")) != "silu"
        || config_json.value("mlp_bias", false)) {
        throw std::invalid_argument("Deepseek requires SiLU and no MLP bias");
    }
    if (!config_json.contains("hidden_act")) {
        config_json["hidden_act"] = "silu";
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

    const size_t kv = model_config->get<size_t>("num_key_value_heads"), dim = model_config->get<size_t>("head_dim");
    if (dim < 2 || heads % kv) {
        throw std::invalid_argument("Deepseek GQA/RoPE geometry is invalid");
    }
    const double eps = model_config->get<double>("rms_norm_eps");
    if (!std::isfinite(eps) || eps <= 0) {
        throw std::invalid_argument("Deepseek norm epsilon is invalid");
    }
    const size_t routed = model_config->get_or<size_t>("n_routed_experts", 0);
    if (routed) {
        const size_t topk = model_config->get<size_t>("num_experts_per_tok");
        if (!topk || topk > routed || model_config->get<int64_t>("moe_intermediate_size") <= 0) {
            throw std::invalid_argument("Deepseek expert/topk dimensions are invalid");
        }
        const size_t shared = model_config->get_or<size_t>("n_shared_experts", 0);
        if (shared && model_config->get<size_t>("moe_intermediate_size") > std::numeric_limits<size_t>::max() / shared) {
            throw std::length_error("Deepseek shared expert size overflow");
        }
    }
    return model_config;
}

} // namespace infinilm::models::deepseek

namespace {

INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    deepseek,
    infinilm::models::deepseek::DeepseekForCausalLM,
    infinilm::models::deepseek::create_deepseek_model_config);
} // namespace
