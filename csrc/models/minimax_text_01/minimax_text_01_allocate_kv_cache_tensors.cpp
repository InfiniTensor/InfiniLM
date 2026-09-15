#include "minimax_text_01_allocate_kv_cache_tensors.hpp"

#include "../../global_state/global_state.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace infinilm::models::minimax_text_01 {

AllocatedHybridCache minimax_text_01_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == model_config) {
        throw std::runtime_error(
            "infinilm::models::minimax_text_01::minimax_text_01_allocate_kv_cache_tensors: "
            "model_config is null");
    }

    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t num_attention_heads = model_config->get<size_t>("num_attention_heads");
    const size_t num_key_value_heads = model_config->get<size_t>("num_key_value_heads");
    const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    // 0 = linear (Lightning) attention, 1 = full attention.
    const std::vector<int> attn_type_list = model_config->get<std::vector<int>>("attn_type_list");
    const auto &dtype{model_config->get_dtype()};
    const auto &kv_cache_dtype{model_config->get_kv_cache_dtype()};

    // Pipeline parallel: each stage only allocates the caches for the decoder
    // layers it owns (same partition formula as MiniMaxText01Model).
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t pp_size = static_cast<size_t>(rank_info.pp_size);
    const size_t pp_stage = static_cast<size_t>(rank_info.pp_stage);
    const size_t local_layer_begin = num_hidden_layers * pp_stage / pp_size;
    const size_t local_layer_end = num_hidden_layers * (pp_stage + 1) / pp_size;

    std::vector<infinicore::Tensor> kv_cache_vec(num_hidden_layers);
    std::vector<infinicore::Tensor> ssm_state_vec(num_hidden_layers);

    // Lightning attention layers only need a recurrent [pool, heads, d, d] state.
    auto allocate_linear_cache = [&](size_t layer_idx, size_t pool_size) {
        ssm_state_vec[layer_idx] = cache::MambaCache::create_layer_ssm_state(
            head_dim, head_dim, num_attention_heads, num_attention_heads,
            dtype, pool_size);
    };

    auto allocate_static_full_cache = [&](size_t layer_idx,
                                          const cache::StaticKVCacheConfig &config) {
        kv_cache_vec[layer_idx] = cache::StaticKVCache::create_layer_kv_cache(
            head_dim, head_dim, num_key_value_heads, num_key_value_heads,
            max_position_embeddings, kv_cache_dtype, config);
    };

    auto allocate_paged_full_cache = [&](size_t layer_idx,
                                         const cache::PagedKVCacheConfig &config) {
        kv_cache_vec[layer_idx] = cache::PagedKVCache::create_layer_kv_cache(
            head_dim, head_dim, num_key_value_heads, num_key_value_heads,
            kv_cache_dtype, config);
    };

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto *static_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_config) {
            throw std::runtime_error(
                "infinilm::models::minimax_text_01::minimax_text_01_allocate_kv_cache_tensors: "
                "invalid static kv cache config type");
        }
        for (size_t layer_idx = local_layer_begin; layer_idx < local_layer_end; ++layer_idx) {
            if (0 == attn_type_list[layer_idx]) {
                allocate_linear_cache(layer_idx, static_config->max_batch_size());
            } else if (1 == attn_type_list[layer_idx]) {
                allocate_static_full_cache(layer_idx, *static_config);
            } else {
                throw std::runtime_error(
                    "infinilm::models::minimax_text_01::minimax_text_01_allocate_kv_cache_tensors: "
                    "unsupported attn_type '"
                    + std::to_string(attn_type_list[layer_idx])
                    + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        ;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_config) {
            throw std::runtime_error(
                "infinilm::models::minimax_text_01::minimax_text_01_allocate_kv_cache_tensors: "
                "invalid paged kv cache config type");
        }
        const size_t state_pool_size = std::max<size_t>(2, paged_config->num_blocks() / 4);
        for (size_t layer_idx = local_layer_begin; layer_idx < local_layer_end; ++layer_idx) {
            if (0 == attn_type_list[layer_idx]) {
                allocate_linear_cache(layer_idx, state_pool_size);
            } else if (1 == attn_type_list[layer_idx]) {
                allocate_paged_full_cache(layer_idx, *paged_config);
            } else {
                throw std::runtime_error(
                    "infinilm::models::minimax_text_01::minimax_text_01_allocate_kv_cache_tensors: "
                    "unsupported attn_type '"
                    + std::to_string(attn_type_list[layer_idx])
                    + "' for layer " + std::to_string(layer_idx));
            }
        }
        break;
    }
    default:
        throw std::runtime_error(
            "infinilm::models::minimax_text_01::minimax_text_01_allocate_kv_cache_tensors: "
            "unsupported attention backend");
    }
    return AllocatedHybridCache{std::move(kv_cache_vec), std::move(ssm_state_vec)};
}

} // namespace infinilm::models::minimax_text_01
