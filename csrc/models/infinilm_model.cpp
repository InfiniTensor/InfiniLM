#include "infinilm_model.hpp"

#include "../cache/kv_cache.hpp"
#include "../engine/forward_context.hpp"
#include "../engine/parallel_state.hpp"

namespace infinilm {

void InfinilmModel::initialize_kv_cache(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> text_config) {

    auto &forward_ctx = engine::get_forward_context();

    auto &kv_cache_vec = forward_ctx.kv_cache_vec;
    kv_cache_vec.clear();

    if (cache_config == nullptr) {
        kv_cache_ = nullptr;
        kv_cache_vec.clear();
        return;
    }
    if (text_config == nullptr) {
        throw std::runtime_error("txt_model_config is not initialized");
    }
    cache_config_ = cache_config->unique_copy();

    const backends::AttentionBackend attention_backend = config::get_current_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::engine::get_tensor_model_parallel_rank_info();
    printf("initialize_kv_cache 222 \n");
    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config_.get());
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error(
                "initialize_kv_cache: expected StaticKVCacheConfig when attention_backend is static-attn; "
                "check cache config type matches the selected attention backend.");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");

        kv_cache_vec.reserve(num_hidden_layers);

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {

            auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                text_config->get_head_dim(),
                text_config->get_head_dim(),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("max_position_embeddings"),
                text_config->get_dtype(),
                *static_kv_cache_config,
                rank_info);
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error("paged_kv_cache_config is not initialized");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
                text_config->get_head_dim(),
                text_config->get_head_dim(),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get_dtype(),
                *paged_kv_cache_config,
                rank_info);
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        auto flash_kv_cache_config = dynamic_cast<const cache::FlashKVCacheConfig *>(cache_config_.get());
        if (nullptr == flash_kv_cache_config) {
            throw std::runtime_error("flash_kv_cache_config is not initialized");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            auto kv_cache = cache::FlashKVCache::create_layer_kv_cache(
                text_config->get_head_dim(),
                text_config->get_head_dim(),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get<size_t>("num_key_value_heads"),
                text_config->get_dtype(),
                *flash_kv_cache_config,
                rank_info);
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
        break;
    }
}

} // namespace infinilm
