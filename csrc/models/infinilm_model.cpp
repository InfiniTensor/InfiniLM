#include "infinilm_model.hpp"

#include "../engine/forward_context.hpp"
#include "../engine/parallel_state.hpp"

namespace infinilm {

void InfinilmModel::initialize_kv_cache(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> text_config) {

    const engine::distributed::RankInfo &rank_info = infinilm::engine::get_tensor_model_parallel_rank_info();

    auto &kv_cache = engine::get_forward_context().kv_cache;

    if (cache_config == nullptr) {
        kv_cache = nullptr;
        cache_config_ = nullptr;
        return;
    }
    if (text_config == nullptr) {
        throw std::runtime_error("txt_model_config is not initialized");
    }
    cache_config_ = cache_config->unique_copy();

    const backends::AttentionBackend attention_backend = config::get_current_infinilm_config().attention_backend;

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config_.get());
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("static_kv_cache_config is not initialized");
        }
        kv_cache = std::make_shared<cache::StaticKVCache>(
            text_config->get_head_dim(),
            text_config->get_head_dim(),
            text_config->get<size_t>("num_key_value_heads"),
            text_config->get<size_t>("num_key_value_heads"),
            text_config->get<size_t>("num_hidden_layers"),
            text_config->get<size_t>("max_position_embeddings"),
            text_config->get_dtype(),
            *static_kv_cache_config,
            rank_info);
        break;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config_.get());
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error("paged_kv_cache_config is not initialized");
        }
        kv_cache = std::make_shared<cache::PagedKVCache>(
            text_config->get_head_dim(),
            text_config->get_head_dim(),
            text_config->get<size_t>("num_key_value_heads"),
            text_config->get<size_t>("num_key_value_heads"),
            text_config->get<size_t>("num_hidden_layers"),
            text_config->get_dtype(),
            *paged_kv_cache_config,
            rank_info);
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        auto flash_kv_cache_config = dynamic_cast<const cache::FlashKVCacheConfig *>(cache_config_.get());
        if (nullptr == flash_kv_cache_config) {
            throw std::runtime_error("flash_kv_cache_config is not initialized");
        }
        kv_cache = std::make_shared<cache::PagedKVCache>(
            text_config->get_head_dim(),
            text_config->get_head_dim(),
            text_config->get<size_t>("num_key_value_heads"),
            text_config->get<size_t>("num_key_value_heads"),
            text_config->get<size_t>("num_hidden_layers"),
            text_config->get_dtype(),
            *flash_kv_cache_config,
            rank_info);
        break;
    }
    default:
        throw std::runtime_error("Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
        break;
    }
}

} // namespace infinilm
