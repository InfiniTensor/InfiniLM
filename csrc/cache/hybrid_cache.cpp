#include "hybrid_cache.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::cache {

HybridCacheTensors allocate_hybrid_cache_tensors(
    const CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == model_config) {
        throw std::runtime_error("allocate_hybrid_cache_tensors: model_config is null");
    }

    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t num_key_value_heads = model_config->get<size_t>("num_key_value_heads");
    const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");

    const size_t linear_conv_kernel_dim = model_config->get<size_t>("linear_conv_kernel_dim");
    const size_t linear_key_head_dim = model_config->get<size_t>("linear_key_head_dim");
    const size_t linear_num_key_heads = model_config->get<size_t>("linear_num_key_heads");
    const size_t linear_num_value_heads = model_config->get<size_t>("linear_num_value_heads");
    const size_t linear_value_head_dim = model_config->get<size_t>("linear_value_head_dim");

    const auto &dtype{model_config->get_dtype()};
    const auto &kv_cache_dtype{model_config->get_kv_cache_dtype()};
    const std::vector<std::string> layer_types = model_config->get<std::vector<std::string>>("layer_types");
    if (layer_types.size() != num_hidden_layers) {
        throw std::runtime_error(
            "allocate_hybrid_cache_tensors: layer_types size must match num_hidden_layers");
    }

    std::vector<infinicore::Tensor> kv_cache_vec;
    std::vector<infinicore::Tensor> conv_state_vec;
    std::vector<infinicore::Tensor> ssm_state_vec;
    kv_cache_vec.reserve(num_hidden_layers);
    conv_state_vec.reserve(num_hidden_layers);
    ssm_state_vec.reserve(num_hidden_layers);

    size_t mamba_state_pool_size = 0;
    auto allocate_linear_attention_cache = [&](size_t layer_idx, size_t pool_size) {
        if (mamba_state_pool_size == 0) {
            mamba_state_pool_size = pool_size;
        } else if (mamba_state_pool_size != pool_size) {
            throw std::runtime_error(
                "allocate_hybrid_cache_tensors: inconsistent mamba state pool size at layer "
                + std::to_string(layer_idx));
        }

        auto conv_state = MambaCache::create_layer_conv_state(
            linear_key_head_dim,
            linear_value_head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_conv_kernel_dim,
            dtype,
            pool_size);
        auto ssm_state = MambaCache::create_layer_ssm_state(
            linear_key_head_dim,
            linear_value_head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            dtype,
            pool_size);

        kv_cache_vec.emplace_back();
        conv_state_vec.push_back(std::move(conv_state));
        ssm_state_vec.push_back(std::move(ssm_state));
    };

    auto allocate_static_full_attention_cache = [&](size_t layer_idx, const StaticKVCacheConfig &config) {
        auto kv_cache = StaticKVCache::create_layer_kv_cache(
            head_dim,
            head_dim,
            num_key_value_heads,
            num_key_value_heads,
            max_position_embeddings,
            kv_cache_dtype,
            config);

        kv_cache_vec.push_back(std::move(kv_cache));
        conv_state_vec.emplace_back();
        ssm_state_vec.emplace_back();
    };

    auto allocate_paged_full_attention_cache = [&](size_t layer_idx, const PagedKVCacheConfig &config) {
        auto kv_cache = PagedKVCache::create_layer_kv_cache(
            head_dim,
            head_dim,
            num_key_value_heads,
            num_key_value_heads,
            kv_cache_dtype,
            config);

        kv_cache_vec.push_back(std::move(kv_cache));
        conv_state_vec.emplace_back();
        ssm_state_vec.emplace_back();
    };

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("allocate_hybrid_cache_tensors: invalid static kv cache config type");
        }

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            const std::string &layer_type = layer_types[layer_idx];
            if ("linear_attention" == layer_type) {
                allocate_linear_attention_cache(layer_idx, static_kv_cache_config->max_batch_size());
            } else if ("full_attention" == layer_type) {
                allocate_static_full_attention_cache(layer_idx, *static_kv_cache_config);
            } else {
                throw std::runtime_error("allocate_hybrid_cache_tensors: unsupported layer_type '" + layer_type + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        ;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error("allocate_hybrid_cache_tensors: invalid paged kv cache config type");
        }
        const size_t mamba_pool_size = std::max<size_t>(2, paged_kv_cache_config->num_blocks() / 4);

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            const std::string &layer_type = layer_types[layer_idx];
            if ("linear_attention" == layer_type) {
                allocate_linear_attention_cache(layer_idx, mamba_pool_size);
            } else if ("full_attention" == layer_type) {
                allocate_paged_full_attention_cache(layer_idx, *paged_kv_cache_config);
            } else {
                throw std::runtime_error("allocate_hybrid_cache_tensors: unsupported layer_type '" + layer_type + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    default:
        throw std::runtime_error("allocate_hybrid_cache_tensors: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
    return HybridCacheTensors{
        std::move(kv_cache_vec),
        std::move(conv_state_vec),
        std::move(ssm_state_vec),
        mamba_state_pool_size};
}

} // namespace infinilm::cache
