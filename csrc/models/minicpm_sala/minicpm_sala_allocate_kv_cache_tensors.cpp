#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "minicpm_sala_for_causal_lm.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::minicpm_sala {

std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> minicpm_sala_allocate_kv_cache_tensors(const cache::CacheConfig *cache_config,
                                                                                                       const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
                                                                                                       const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == text_config) {
        throw std::runtime_error("infinilm::models::minicpm_sala::minicpm_sala_allocate_kv_cache_tensors: text_config is null");
    }

    std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> kv_cache_vec;

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::models::minicpm_sala::minicpm_sala_allocate_kv_cache_tensors: invalid static kv cache config type");
        }
        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        const size_t head_dim = text_config->get<size_t>("head_dim");
        const size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
        const size_t max_position_embeddings = text_config->get<size_t>("max_position_embeddings");

        const auto &dtype{text_config->get_dtype()};
        std::vector<std::string> mixer_types = text_config->get<std::vector<std::string>>("mixer_types");
        size_t current_layer_head_dim, current_layer_num_key_value_heads;
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            std::string mixer_type = mixer_types[layer_idx];
            if ("minicpm4" == mixer_type) {
                current_layer_head_dim = head_dim;
                current_layer_num_key_value_heads = num_key_value_heads;
            } else if ("lightning" == mixer_type || "lightning_attn" == mixer_type || "lightning-attn" == mixer_type) {
                current_layer_head_dim = text_config->get<size_t>("lightning_head_dim");
                current_layer_num_key_value_heads = text_config->get<size_t>("lightning_nkv");
            } else {
                throw std::runtime_error("infinilm::models::minicpm_sala::minicpm_sala_allocate_kv_cache_tensors: unsupported mixer_type '" + mixer_type + "' for layer " + std::to_string(layer_idx));
            }
            auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                current_layer_head_dim,
                current_layer_head_dim,
                current_layer_num_key_value_heads,
                current_layer_num_key_value_heads,
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
                "infinilm::models::minicpm_sala::minicpm_sala_allocate_kv_cache_tensors: invalid paged kv cache config type");
        }

        const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
        kv_cache_vec.reserve(num_hidden_layers);

        const size_t head_dim = text_config->get<size_t>("head_dim");
        const size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
        const auto &dtype{text_config->get_dtype()};
        std::vector<std::string> mixer_types = text_config->get<std::vector<std::string>>("mixer_types");
        size_t current_layer_head_dim, current_layer_num_key_value_heads;
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            std::string mixer_type = mixer_types[layer_idx];
            if ("minicpm4" == mixer_type) {
                current_layer_head_dim = head_dim;
                current_layer_num_key_value_heads = num_key_value_heads;
            } else if ("lightning" == mixer_type || "lightning_attn" == mixer_type || "lightning-attn" == mixer_type) {
                current_layer_head_dim = text_config->get<size_t>("lightning_head_dim");
                current_layer_num_key_value_heads = text_config->get<size_t>("lightning_nkv");
            } else {
                throw std::runtime_error("infinilm::models::minicpm_sala::minicpm_sala_allocate_kv_cache_tensors: unsupported mixer_type '" + mixer_type + "' for layer " + std::to_string(layer_idx));
            }
            auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
                current_layer_head_dim,
                current_layer_head_dim,
                current_layer_num_key_value_heads,
                current_layer_num_key_value_heads,
                dtype,
                *paged_kv_cache_config);
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    default:
        throw std::runtime_error("infinilm::models::minicpm_sala::minicpm_sala_allocate_kv_cache_tensors: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
    return kv_cache_vec;
}

} // namespace infinilm::models::minicpm_sala
