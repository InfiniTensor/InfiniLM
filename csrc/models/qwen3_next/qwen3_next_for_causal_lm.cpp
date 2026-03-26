#include "qwen3_next_for_causal_lm.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/infinilm_config.hpp"
#include "../../engine/forward_context.hpp"
#include "../../engine/parallel_state.hpp"
#include "../models_registry.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextForCausalLM::Qwen3NextForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {

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
    auto &model_config = infinilm::config::get_current_infinilm_config().model_config;

    initialize_kv_cache_(cache_config, model_config);
}

void Qwen3NextForCausalLM::initialize_kv_cache_(const cache::CacheConfig *cache_config,
                                                const std::shared_ptr<infinilm::config::ModelConfig> &text_config) {

    auto &forward_ctx = engine::get_forward_context();
    auto &kv_cache_vec = forward_ctx.kv_cache_vec;
    kv_cache_vec.clear();

    if (nullptr == cache_config) {
        return;
    }

    std::shared_ptr<infinilm::config::ModelConfig> effective_text_config = text_config;
    if (nullptr == effective_text_config) {
        effective_text_config = infinilm::config::get_current_infinilm_config().model_config;
    }
    if (nullptr == effective_text_config) {
        throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: text_config is null");
    }

    const backends::AttentionBackend attention_backend = infinilm::config::get_current_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::engine::get_tensor_model_parallel_rank_info();

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: invalid static kv cache config type");
        }
        const size_t num_hidden_layers = effective_text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        const size_t head_dim = effective_text_config->get<size_t>("head_dim");
        const size_t num_key_value_heads = effective_text_config->get<size_t>("num_key_value_heads");
        const size_t max_position_embeddings = effective_text_config->get<size_t>("max_position_embeddings");

        const auto &dtype{effective_text_config->get_dtype()};
        const std::vector<std::string> layer_types = effective_text_config->get<std::vector<std::string>>("layer_types");

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            const std::string &layer_type = layer_types[layer_idx];

            if ("linear_attention" == layer_type) {
                kv_cache_vec.emplace_back();
            } else if ("full_attention" == layer_type) {
                auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                    head_dim,
                    head_dim,
                    num_key_value_heads,
                    num_key_value_heads,
                    max_position_embeddings,
                    dtype,
                    *static_kv_cache_config,
                    rank_info);

                kv_cache_vec.push_back(kv_cache);
            } else {
                throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextForCausalLM::initialize_kv_cache_: unsupported layer_type '" + layer_type + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    default:
        throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
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

    if (!config_json.contains("qk_norm")) {
        config_json["qk_norm"] = true;
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
