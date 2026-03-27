#include "qwen3_next_for_causal_lm.hpp"
#include "../../config/infinilm_config.hpp"
#include "../../engine/forward_context.hpp"
#include "../models_registry.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextForCausalLM::Qwen3NextForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    model_config_ = model_config;
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output Qwen3NextForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void Qwen3NextForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto &kv_cache_vec = engine::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::config::get_current_infinilm_config().attention_backend;

    auto new_kv_cache_vec = qwen3_next_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend);
    kv_cache_vec = std::move(new_kv_cache_vec);
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_next" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_next::create_qwen3_next_model_config: model_type is not qwen3_next");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("layer_types")) {
        size_t full_attention_interval = model_config->get<size_t>("full_attention_interval");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; i++) {
            layer_types.push_back(bool((i + 1) % full_attention_interval) ? "linear_attention" : "full_attention");
        }
        config_json["layer_types"] = layer_types;
    }

    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_next

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_next,
    infinilm::models::qwen3_next::Qwen3NextForCausalLM,
    infinilm::models::qwen3_next::create_qwen3_next_model_config);
} // namespace
