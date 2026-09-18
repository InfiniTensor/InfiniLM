#include "granitemoehybrid_allocate_kv_cache_tensors.hpp"

#include "../../global_state/global_state.hpp"

#include "infinicore/context/context.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::granitemoehybrid {

GraniteMoeHybridAllocatedCache granitemoehybrid_allocate_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == text_config) {
        throw std::runtime_error("infinilm::models::granitemoehybrid::granitemoehybrid_allocate_cache_tensors: text_config is null");
    }

    const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
    const size_t head_dim = text_config->get_head_dim();
    const size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");

    const size_t hidden_size = text_config->get<size_t>("hidden_size");
    const size_t mamba_expand = text_config->get_or<size_t>("mamba_expand", 2);
    const size_t mamba_n_groups = text_config->get_or<size_t>("mamba_n_groups", 1);
    const size_t mamba_d_state = text_config->get<size_t>("mamba_d_state");
    const size_t mamba_d_conv = text_config->get<size_t>("mamba_d_conv");

    const auto &dtype = text_config->get_dtype();
    const auto &kv_cache_dtype = text_config->get_kv_cache_dtype();
    const std::vector<std::string> layer_types = text_config->get<std::vector<std::string>>("layer_types");

    std::vector<infinicore::Tensor> kv_cache_vec;
    std::vector<infinicore::Tensor> conv_state_vec;
    kv_cache_vec.reserve(num_hidden_layers);
    conv_state_vec.reserve(num_hidden_layers);

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t local_mamba_groups = mamba_n_groups >= static_cast<size_t>(rank_info.tp_size)
                                        ? mamba_n_groups / rank_info.tp_size
                                        : 1;
    const size_t mamba_conv_dim = mamba_expand * hidden_size / rank_info.tp_size + 2 * local_mamba_groups * mamba_d_state;

    auto allocate_mamba_cache = [&](size_t pool_size) {
        auto conv_state = infinicore::Tensor::zeros(
            {pool_size, mamba_conv_dim, mamba_d_conv - 1},
            dtype,
            rank_info.device);
        kv_cache_vec.emplace_back();
        conv_state_vec.push_back(std::move(conv_state));
    };

    auto allocate_static_attention_cache = [&](const cache::StaticKVCacheConfig &config) {
        auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
            head_dim,
            head_dim,
            num_key_value_heads,
            num_key_value_heads,
            text_config->get<size_t>("max_position_embeddings"),
            kv_cache_dtype,
            config);

        kv_cache_vec.push_back(std::move(kv_cache));
        conv_state_vec.emplace_back();
    };

    auto allocate_paged_attention_cache = [&](const cache::PagedKVCacheConfig &config) {
        auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
            head_dim,
            head_dim,
            num_key_value_heads,
            num_key_value_heads,
            kv_cache_dtype,
            config);

        kv_cache_vec.push_back(std::move(kv_cache));
        conv_state_vec.emplace_back();
    };

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        const auto *static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::models::granitemoehybrid::granitemoehybrid_allocate_cache_tensors: invalid static kv cache config type");
        }

        const size_t mamba_pool_size = std::max<size_t>(2, static_kv_cache_config->max_batch_size() + 1);
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            const std::string &layer_type = layer_types[layer_idx];
            if ("mamba" == layer_type) {
                allocate_mamba_cache(mamba_pool_size);
            } else if ("attention" == layer_type) {
                allocate_static_attention_cache(*static_kv_cache_config);
            } else {
                throw std::runtime_error("infinilm::models::granitemoehybrid::granitemoehybrid_allocate_cache_tensors: unsupported layer_type '" + layer_type + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        ;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        const auto *paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error("infinilm::models::granitemoehybrid::granitemoehybrid_allocate_cache_tensors: invalid paged kv cache config type");
        }
        const size_t mamba_pool_size = std::max<size_t>(2, paged_kv_cache_config->num_blocks() / 4);

        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            const std::string &layer_type = layer_types[layer_idx];
            if ("mamba" == layer_type) {
                allocate_mamba_cache(mamba_pool_size);
            } else if ("attention" == layer_type) {
                allocate_paged_attention_cache(*paged_kv_cache_config);
            } else {
                throw std::runtime_error("infinilm::models::granitemoehybrid::granitemoehybrid_allocate_cache_tensors: unsupported layer_type '" + layer_type + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    default:
        throw std::runtime_error("infinilm::models::granitemoehybrid::granitemoehybrid_allocate_cache_tensors: unsupported attention backend " + std::to_string(static_cast<int>(attention_backend)));
    }
    infinicore::context::syncStream();
    return GraniteMoeHybridAllocatedCache{
        std::move(kv_cache_vec),
        std::move(conv_state_vec)};
}

} // namespace infinilm::models::granitemoehybrid
