#include "ernie4_5_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::ernie4_5_vl {
namespace {

size_t get_first_size(const nlohmann::json &config, const char *key, size_t default_value) {
    if (!config.contains(key) || config.at(key).is_null()) {
        return default_value;
    }
    const auto &value = config.at(key);
    if (value.is_array()) {
        return value.empty() ? default_value : value.at(0).get<size_t>();
    }
    return value.get<size_t>();
}

void normalize_ernie_config(nlohmann::json &config_json) {
    if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
        config_json["dtype"] = config_json["torch_dtype"];
    }
    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = config_json["hidden_size"].get<size_t>() / config_json["num_attention_heads"].get<size_t>();
    }
    if (!config_json.contains("partial_rotary_factor")) {
        config_json["partial_rotary_factor"] = 1.0;
    }
    if (!config_json.contains("compression_ratio")) {
        config_json["compression_ratio"] = 1.0;
    }
    if (!config_json.contains("use_flash_attention")) {
        config_json["use_flash_attention"] = true;
    }
    if (!config_json.contains("attention_probs_dropout_prob")) {
        config_json["attention_probs_dropout_prob"] = 0.0;
    }
    if (!config_json.contains("hidden_dropout_prob")) {
        config_json["hidden_dropout_prob"] = 0.0;
    }
    if (!config_json.contains("mlp_bias")) {
        config_json["mlp_bias"] = config_json.value("use_bias", false);
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = config_json.value("use_bias", false);
    }
    if (!config_json.contains("norm_topk_prob")) {
        config_json["norm_topk_prob"] = config_json.value("moe_norm_gate_logits", true);
    }
    if (!config_json.contains("num_experts")) {
        config_json["num_experts"] = get_first_size(config_json, "moe_num_experts", 0);
    }
    if (!config_json.contains("num_experts_per_tok")) {
        config_json["num_experts_per_tok"] = config_json.value("moe_k", 1);
    }
    if (!config_json.contains("use_moe")) {
        const size_t num_experts = config_json.value("num_experts", 0);
        config_json["use_moe"] = num_experts > 0;
    }
    if (!config_json.contains("moe_dropout_prob")) {
        config_json["moe_dropout_prob"] = 0.0;
    }
    if (!config_json.contains("moe_reverse_token_drop")) {
        config_json["moe_reverse_token_drop"] = false;
    }
    if (!config_json.contains("moe_group")) {
        config_json["moe_group"] = "world";
    }
    if (!config_json.contains("moe_all_to_all_dropout")) {
        config_json["moe_all_to_all_dropout"] = 0.0;
    }
}

} // namespace

Ernie45ForConditionalGeneration::Ernie45ForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                                 const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();
    auto &config_json = model_config->get_config_json();
    if (config_json.contains("vision_config") && config_json["vision_config"].is_object()) {
        INFINICORE_NN_MODULE_INIT(vision_model, config_json["vision_config"], dtype, device);
    }

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output Ernie45ForConditionalGeneration::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = (input.pixel_values.has_value() && !input.pixel_values.value().empty())
                           ? model_->forward(input, vision_model_.get())
                           : model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void Ernie45ForConditionalGeneration::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();

    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    forward_context.kv_cache_vec = std::move(default_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("ernie4_5_moe_vl" != model_type) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::create_ernie4_5_moe_vl_model_config: model_type is not ernie4_5_moe_vl");
    }
    normalize_ernie_config(model_config->get_config_json());
    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);
    return model_config;
}

} // namespace infinilm::models::ernie4_5_vl

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    ernie4_5_moe_vl,
    infinilm::models::ernie4_5_vl::Ernie45ForConditionalGeneration,
    infinilm::models::ernie4_5_vl::create_ernie4_5_moe_vl_model_config);
} // namespace
