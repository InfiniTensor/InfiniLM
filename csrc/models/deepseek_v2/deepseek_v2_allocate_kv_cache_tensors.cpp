#include "deepseek_v2_for_causal_lm.hpp"

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v2 {

std::vector<infinicore::Tensor> deepseek_v2_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    if (text_config == nullptr) {
        throw std::runtime_error("deepseek_v2_allocate_kv_cache_tensors: text_config is null");
    }
    if (attention_backend == backends::AttentionBackend::STATIC_ATTN) {
        throw std::runtime_error("DeepSeek V2 requires the vLLM-style paged MLA cache");
    }
    auto paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (paged_config == nullptr) {
        throw std::runtime_error("deepseek_v2_allocate_kv_cache_tensors: expected paged KV cache config");
    }
    if (paged_config->block_size() != 16) {
        throw std::runtime_error(
            "deepseek_v2_allocate_kv_cache_tensors: the current Iluvatar MLA SO requires block_size=16");
    }

    const size_t num_layers = text_config->get<size_t>("num_hidden_layers");
    const size_t cache_dim = text_config->get<size_t>("kv_lora_rank")
                           + text_config->get<size_t>("qk_rope_head_dim");
    const auto &dtype = text_config->get_kv_cache_dtype();
    const auto &device = global_state::get_tensor_model_parallel_rank_info().device;
    std::vector<infinicore::Tensor> caches;
    caches.reserve(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        auto cache = infinicore::Tensor::empty(
            {paged_config->num_blocks(), paged_config->block_size(), cache_dim}, dtype, device);
        set_zeros(cache);
        caches.push_back(cache);
    }
    infinicore::context::syncStream();
    return caches;
}

} // namespace infinilm::models::deepseek_v2
