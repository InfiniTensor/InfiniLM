#include "kimi_k3_allocate_cache.hpp"

#include "kimi_k3_pipeline_partition.hpp"

#include "runtime/runtime.hpp"
#include "../../global_state/global_state.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <vector>

namespace infinilm::models::kimi_k3 {

qwen3_next::AllocatedHybridCache kimi_k3_allocate_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank_info,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    backends::AttentionBackend attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    cache::validate_cache_runtime(runtime, rank_info);
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    if (attention_backend == backends::AttentionBackend::STATIC_ATTN) {
        throw std::runtime_error("Kimi K3 does not support static attention");
    }
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t num_heads = model_config->get<size_t>("num_attention_heads");
    const auto &linear = model_config->get_config_json().at("linear_attn_config");
    const size_t linear_head_dim = linear.at("head_dim").get<size_t>();
    const size_t linear_num_heads = linear.at("num_heads").get<size_t>();
    const size_t conv_kernel = linear.at("short_conv_kernel_size").get<size_t>();
    if(conv_kernel<2) throw std::invalid_argument("K3 conv state requires kernel >=2");
    const auto kda_layers = linear.at("kda_layers").get<std::vector<size_t>>();
    std::vector<bool> is_kda(num_layers, false);
    for (const size_t one_based_layer_idx : kda_layers) {
        is_kda.at(one_based_layer_idx - 1) = true;
    }

    runtime.assertOwnerThread();
    if(!tp_size || linear_num_heads%tp_size) throw std::invalid_argument("K3 state head partition invalid");
    const auto [local_begin, local_end] = kimi_k3_pipeline_layer_range(
        num_layers,
        rank_info.pp_size,
        rank_info.pp_stage);
    std::vector<infinilm::runtime::Tensor> kv(num_layers);
    std::vector<infinilm::runtime::Tensor> conv(num_layers);
    std::vector<infinilm::runtime::Tensor> recurrent(num_layers);

    auto allocate_kda = [&](size_t layer_idx, size_t pool_size) {
        const size_t heads=linear_num_heads/tp_size;
        conv[layer_idx]=infinilm::runtime::Tensor::empty(runtime,{pool_size,3*heads*linear_head_dim,conv_kernel-1},model_config->get_dtype());
        recurrent[layer_idx]=infinilm::runtime::Tensor::empty(runtime,{pool_size,heads,linear_head_dim,linear_head_dim},model_config->get_dtype());
        runtime.setDeviceMemory(conv[layer_idx]->data(),0,conv[layer_idx]->nbytes());
        runtime.setDeviceMemory(recurrent[layer_idx]->data(),0,recurrent[layer_idx]->nbytes());
        runtime.syncStream(); // preserve zero-row initialization before publication

    };

    const auto *config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (config == nullptr) {
        throw std::runtime_error("Kimi K3 paged attention requires PagedKVCacheConfig");
    }
    const size_t state_pool_size = std::max<size_t>(2, config->num_blocks() / 4);
    try {
    for (size_t i = local_begin; i < local_end; ++i) {
        if (is_kda[i]) {
            allocate_kda(i, state_pool_size);
        } else {
            kv[i] = cache::PagedKVCache::create_layer_kv_cache(
                runtime, rank_info,
                head_dim, head_dim, num_heads, num_heads,
                model_config->get_kv_cache_dtype(), *config);
        }
    }
    } catch (...) {
        // Local vectors still retain all newly allocated backing here.
        const auto primary=std::current_exception();
        try { runtime.syncStream(); }
        catch (...) { std::terminate(); } // no unwind/reuse after a failed drain
        std::rethrow_exception(primary);
    }
    return {std::move(kv), std::move(conv), std::move(recurrent)};
}

} // namespace infinilm::models::kimi_k3
