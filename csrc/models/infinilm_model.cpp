#include "infinilm_model.hpp"

#include "../backends/attention_backends.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/infinilm_config.hpp"
#include "../engine/distributed/distributed.hpp"
#include "../engine/forward_context.hpp"
#include "../engine/parallel_state.hpp"

#include <stdexcept>

namespace infinilm {

void InfinilmModel::initialize_kv_cache(const cache::CacheConfig *cache_config,
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

        size_t head_dim = effective_text_config->get<size_t>("head_dim");
        size_t num_key_value_heads = effective_text_config->get<size_t>("num_key_value_heads");
        size_t max_position_embeddings = effective_text_config->get<size_t>("max_position_embeddings");
        const auto &dtype{effective_text_config->get_dtype()};

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
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
        }
        break;
    }
    case backends::AttentionBackend::PAGED_ATTN: {

        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error(
                "initialize_kv_cache: expected PagedKVCacheConfig when attention_backend is paged-attn or flash-attn");
        }

        const size_t num_hidden_layers = effective_text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        size_t head_dim = effective_text_config->get<size_t>("head_dim");
        size_t num_key_value_heads = effective_text_config->get<size_t>("num_key_value_heads");
        const auto &dtype{effective_text_config->get_dtype()};

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
                head_dim,
                head_dim,
                num_key_value_heads,
                num_key_value_heads,
                dtype,
                *paged_kv_cache_config,
                rank_info);
            kv_cache_vec.push_back(kv_cache);
        }

        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: flash-attn is not supported");

        break;
    }
    default:
        throw std::runtime_error("infinilm::InfinilmModel::initialize_kv_cache: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
}

} // namespace infinilm
