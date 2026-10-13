#include "../../layers/rotary_embedding/rope.hpp"
#include "ernie4_5_for_causal_lm.hpp"

#include "../../global_state/forward_context.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

#include <exception>

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

Ernie45ForConditionalGeneration::Ernie45ForConditionalGeneration(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    try {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();
    auto &config_json = model_config->get_config_json();
    if (config_json.contains("vision_config") && config_json["vision_config"].is_object()) {
            vision_model_ = std::make_shared<std::remove_reference_t<decltype(*vision_model_)>>(runtime, config_json["vision_config"], dtype);
            this->bind_child("vision_model", vision_model_);
    }

        model_ = std::make_shared<std::remove_reference_t<decltype(*model_)>>(runtime, model_config, rank_info, attention_backend, ep_config);
        this->bind_child("model", model_);
        lm_head_ = std::make_shared<std::remove_reference_t<decltype(*lm_head_)>>(runtime, hidden_size, vocab_size, false, dtype);
        this->bind_child("lm_head", lm_head_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::InfinilmModel::Output Ernie45ForConditionalGeneration::forward(
    const infinilm::InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto hidden_states = model_->forward(input, forward_context, vision_model_.get());
    auto logits = lm_head_->forward(forward_context.require_runtime(), hidden_states);
    return {logits, hidden_states};
}

void Ernie45ForConditionalGeneration::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (cache_config == nullptr) {
        cache_config_.reset();
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank);
    cache_config_ = cache_config->unique_copy();

    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();

    forward_context.kv_cache_vec = std::move(default_allocate_kv_cache_tensors(runtime, rank, cache_config, model_config_, attention_backend));
}

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("ernie4_5_moe_vl" != model_type) {
        throw std::runtime_error("infinilm::models::ernie4_5_vl::create_ernie4_5_moe_vl_model_config: model_type is not ernie4_5_moe_vl");
    }
    if (model_config->get<int64_t>("hidden_size") <= 0 || model_config->get<int64_t>("num_attention_heads") <= 0)
        throw std::invalid_argument("Ernie45 construction requires positive hidden_size and num_attention_heads");
    normalize_ernie_config(model_config->get_config_json());
    model_config->set_rope_algo(infinilm::layers::RoPE::Algo::GPT_J);
    return model_config;
}

} // namespace infinilm::models::ernie4_5_vl

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    ernie4_5_moe_vl,
    infinilm::models::ernie4_5_vl::Ernie45ForConditionalGeneration,
    infinilm::models::ernie4_5_vl::create_ernie4_5_moe_vl_model_config);
} // namespace
