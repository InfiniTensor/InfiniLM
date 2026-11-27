#include "gemma4_allocate_kv_cache_tensors.hpp"

#include "../../parallel/topology.hpp"
#include "gemma4_config.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::models::gemma4 {

std::vector<infinilm::runtime::Tensor> gemma4_allocate_kv_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank_info,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    cache::validate_backend_cache_config(cache_config, attention_backend);
    cache::validate_cache_runtime(runtime, rank_info);
    if (!model_config) {
        throw std::invalid_argument("gemma4_allocate_kv_cache_tensors: model_config is null");
    }

    const auto spec = parse_gemma4_text_spec(model_config);
    const auto kv_dtype = model_config->get_kv_cache_dtype();
    const parallel::Topology topology(static_cast<size_t>(rank_info.tp_size), rank_info.pp_size, rank_info.pp_stage);
    const auto [local_begin, local_end] = topology.layer_range(spec.num_hidden_layers);

    std::vector<infinilm::runtime::Tensor> kv_cache_vec(spec.num_hidden_layers);
    try {
        switch (attention_backend) {
        case backends::AttentionBackend::STATIC_ATTN: {
            const auto *static_cfg = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
            if (!static_cfg) {
                throw std::runtime_error("gemma4_allocate_kv_cache_tensors: invalid StaticKVCacheConfig");
            }
            for (size_t i = local_begin; i < local_end; ++i) {
                const auto &ls = spec.layers[i];
                if (ls.is_kv_shared) {
                    // Shared layers read from kv_cache_vec[ls.kv_source_layer_idx]; zero extra memory!
                    continue;
                }
                kv_cache_vec[i] = cache::StaticKVCache::create_layer_kv_cache(
                    runtime, rank_info,
                    ls.head_dim, ls.head_dim,
                    ls.num_kv_heads, ls.num_kv_heads,
                    spec.max_position_embeddings,
                    kv_dtype, *static_cfg);
            }
            break;
        }
        case backends::AttentionBackend::FLASH_ATTN:
        case backends::AttentionBackend::PAGED_ATTN: {
            const auto *paged_cfg = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
            if (!paged_cfg) {
                throw std::runtime_error("gemma4_allocate_kv_cache_tensors: invalid PagedKVCacheConfig");
            }
            for (size_t i = local_begin; i < local_end; ++i) {
                const auto &ls = spec.layers[i];
                if (ls.is_kv_shared) {
                    continue;
                }
                kv_cache_vec[i] = cache::PagedKVCache::create_layer_kv_cache(
                    runtime, rank_info,
                    ls.head_dim, ls.head_dim,
                    ls.num_kv_heads, ls.num_kv_heads,
                    kv_dtype, *paged_cfg);
            }
            runtime.syncStream();
            break;
        }
        default:
            throw std::runtime_error("gemma4_allocate_kv_cache_tensors: unsupported attention backend");
        }
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
    return kv_cache_vec;
}

} // namespace infinilm::models::gemma4
