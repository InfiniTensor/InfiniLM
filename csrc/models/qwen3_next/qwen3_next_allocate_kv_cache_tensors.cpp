#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "qwen3_next_for_causal_lm.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_next {

std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> qwen3_next_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == text_config) {
        throw std::runtime_error("infinilm::models::qwen3_next::qwen3_next_allocate_kv_cache_tensors: text_config is null");
    }

    std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> kv_cache_vec;
    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::models::qwen3_next::qwen3_next_allocate_kv_cache_tensors: invalid static kv cache config type");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        const size_t head_dim = text_config->get<size_t>("head_dim");
        const size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
        const size_t max_position_embeddings = text_config->get<size_t>("max_position_embeddings");
        const auto &dtype{text_config->get_dtype()};
        const std::vector<std::string> layer_types = text_config->get<std::vector<std::string>>("layer_types");

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
                    *static_kv_cache_config);
                kv_cache_vec.push_back(kv_cache);
            } else {
                throw std::runtime_error("infinilm::models::qwen3_next::qwen3_next_allocate_kv_cache_tensors: unsupported layer_type '" + layer_type + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        ;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error("infinilm::models::qwen3_next::qwen3_next_allocate_kv_cache_tensors: invalid paged kv cache config type");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        const size_t head_dim = text_config->get<size_t>("head_dim");
        const size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
        const auto &dtype{text_config->get_dtype()};
        const std::vector<std::string> layer_types = text_config->get<std::vector<std::string>>("layer_types");

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            const std::string &layer_type = layer_types[layer_idx];
            if ("linear_attention" == layer_type) {
                kv_cache_vec.emplace_back();
            } else if ("full_attention" == layer_type) {
                auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
                    head_dim,
                    head_dim,
                    num_key_value_heads,
                    num_key_value_heads,
                    dtype,
                    *paged_kv_cache_config);
                kv_cache_vec.push_back(kv_cache);
            } else {
                throw std::runtime_error("infinilm::models::qwen3_next::qwen3_next_allocate_kv_cache_tensors: unsupported layer_type '" + layer_type + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    default:
        throw std::runtime_error("infinilm::models::qwen3_next::qwen3_next_allocate_kv_cache_tensors: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
    return kv_cache_vec;
}

} // namespace infinilm::models::qwen3_next
