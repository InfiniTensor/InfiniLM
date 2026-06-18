#include "deepseek_v2_for_causal_lm.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v2 {

std::vector<infinicore::Tensor> deepseek_v2_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == text_config) {
        throw std::runtime_error("infinilm::models::deepseek_v2::deepseek_v2_allocate_kv_cache_tensors: text_config is null");
    }

    const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
    const size_t kv_lora_rank = text_config->get<size_t>("kv_lora_rank");
    const size_t qk_rope_head_dim = text_config->get<size_t>("qk_rope_head_dim");
    const size_t mla_head_dim = kv_lora_rank + qk_rope_head_dim;
    constexpr size_t num_mla_kv_heads = 1;
    const auto &dtype = text_config->get_kv_cache_dtype();

    std::vector<infinicore::Tensor> kv_cache_vec;
    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::models::deepseek_v2::deepseek_v2_allocate_kv_cache_tensors: invalid static kv cache config type");
        }
        const size_t max_position_embeddings = text_config->get<size_t>("max_position_embeddings");
        kv_cache_vec.reserve(num_hidden_layers);
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                mla_head_dim,
                mla_head_dim,
                num_mla_kv_heads,
                num_mla_kv_heads,
                max_position_embeddings,
                dtype,
                *static_kv_cache_config);
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        ;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error("infinilm::models::deepseek_v2::deepseek_v2_allocate_kv_cache_tensors: invalid paged kv cache config type");
        }
        const size_t mla_cache_dim = mla_head_dim + kv_lora_rank;
        const auto &device = global_state::get_tensor_model_parallel_rank_info().device;
        kv_cache_vec.reserve(num_hidden_layers);
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            auto kv_cache = infinicore::Tensor::empty(
                {paged_kv_cache_config->num_blocks(), num_mla_kv_heads, paged_kv_cache_config->block_size(), mla_cache_dim},
                dtype,
                device);
            set_zeros(kv_cache);
            infinicore::context::syncStream();
            kv_cache_vec.push_back(kv_cache);
        }
        break;
    }
    default:
        throw std::runtime_error("infinilm::models::deepseek_v2::deepseek_v2_allocate_kv_cache_tensors: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
    return kv_cache_vec;
}

} // namespace infinilm::models::deepseek_v2
