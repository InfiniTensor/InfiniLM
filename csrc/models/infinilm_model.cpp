#include "infinilm_model.hpp"
#include "../backends/attention_backends.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/infinilm_config.hpp"
#include "../engine/forward_context.hpp"

#include <stdexcept>

namespace infinilm {

void InfinilmModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
        engine::get_forward_context().kv_cache_vec.clear();
        return;
    }
    cache_config_ = cache_config->unique_copy();
    auto &kv_cache_vec = engine::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::config::get_current_infinilm_config().attention_backend;
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend));
}

std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> InfinilmModel::default_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == text_config) {
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: text_config is null");
    }

    std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> kv_cache_vec;
    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: invalid static kv cache config type");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        size_t head_dim = text_config->get<size_t>("head_dim");
        size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
        size_t max_position_embeddings = text_config->get<size_t>("max_position_embeddings");
        const auto &dtype{text_config->get_dtype()};
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                head_dim,
                head_dim,
                num_key_value_heads,
                num_key_value_heads,
                max_position_embeddings,
                dtype,
                *static_kv_cache_config);
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error(
                "infinilm::InfinilmModel::default_allocate_kv_cache_tensors: invalid paged kv cache config type");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        size_t head_dim = text_config->get<size_t>("head_dim");
        size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
        const auto &dtype{text_config->get_dtype()};
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
                head_dim,
                head_dim,
                num_key_value_heads,
                num_key_value_heads,
                dtype,
                *paged_kv_cache_config);
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: flash-attn is not supported");
        break;
    }
    default:
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
    return kv_cache_vec;
}

} // namespace infinilm
