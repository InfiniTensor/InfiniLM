#include "lfm2_allocate_cache.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/context/context.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::lfm2 {

AllocatedLfm2Cache allocate_lfm2_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    backends::AttentionBackend attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    if (model_config == nullptr) {
        throw std::runtime_error("allocate_lfm2_cache_tensors: model config is null");
    }

    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    const size_t max_positions =
        model_config->get<size_t>("max_position_embeddings");
    const size_t state_length = model_config->get<size_t>("conv_L_cache") - 1;
    const auto layer_types =
        model_config->get<std::vector<std::string>>("layer_types");
    const auto dtype = model_config->get_dtype();
    const auto kv_dtype = model_config->get_kv_cache_dtype();

    const auto &rank_info =
        infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t pp_size = static_cast<size_t>(rank_info.pp_size);
    const size_t pp_stage = static_cast<size_t>(rank_info.pp_stage);
    const size_t local_begin = num_layers * pp_stage / pp_size;
    const size_t local_end = num_layers * (pp_stage + 1) / pp_size;

    AllocatedLfm2Cache result;
    result.kv_cache_tensors.resize(num_layers);
    result.conv_state_tensors.resize(num_layers);
    const auto device = infinicore::context::getDevice();

    if (attention_backend == backends::AttentionBackend::STATIC_ATTN) {
        auto config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (config == nullptr) {
            throw std::runtime_error(
                "allocate_lfm2_cache_tensors: invalid static cache config");
        }
        // Row 0 is immutable zero history; static scheduling uses row 1 for
        // its current request. A new prefill must not read the previous
        // request's terminal ShortConv state.
        const size_t state_pool_size = config->max_batch_size() + 1;
        for (size_t i = local_begin; i < local_end; ++i) {
            if (layer_types.at(i) == "full_attention") {
                result.kv_cache_tensors[i] = cache::StaticKVCache::create_layer_kv_cache(
                    head_dim, head_dim, num_kv_heads, num_kv_heads,
                    max_positions, kv_dtype, *config);
            } else if (layer_types.at(i) == "short_conv") {
                result.conv_state_tensors[i] = infinicore::Tensor::zeros(
                    {state_pool_size, hidden_size, state_length},
                    dtype, device);
            }
        }
        return result;
    }

    if (attention_backend == backends::AttentionBackend::PAGED_ATTN
        || attention_backend == backends::AttentionBackend::FLASH_ATTN) {
        auto config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (config == nullptr) {
            throw std::runtime_error(
                "allocate_lfm2_cache_tensors: invalid paged cache config");
        }
        const size_t state_pool_size =
            std::max<size_t>(2, config->num_blocks() / 4);
        for (size_t i = local_begin; i < local_end; ++i) {
            if (layer_types.at(i) == "full_attention") {
                result.kv_cache_tensors[i] = cache::PagedKVCache::create_layer_kv_cache(
                    head_dim, head_dim, num_kv_heads, num_kv_heads,
                    kv_dtype, *config);
            } else if (layer_types.at(i) == "short_conv") {
                result.conv_state_tensors[i] = infinicore::Tensor::zeros(
                    {state_pool_size, hidden_size, state_length},
                    dtype, device);
            }
        }
        infinicore::context::syncStream();
        return result;
    }

    throw std::runtime_error(
        "allocate_lfm2_cache_tensors: unsupported attention backend");
}

} // namespace infinilm::models::lfm2
