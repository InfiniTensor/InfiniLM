#include "infinilm_model.hpp"
#include "../cache/kv_cache.hpp"
#include "../global_state/global_state.hpp"
#include "../utils.hpp"
#include <stdexcept>

namespace infinilm {

void InfinilmModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
        global_state::get_forward_context().kv_cache_vec.clear();
        global_state::get_forward_context().kv_scale_vec.clear();
        return;
    }
    cache_config_ = cache_config->unique_copy();
    auto &kv_cache_vec = global_state::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    kv_cache_vec = std::move(default_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend));
}

std::vector<infinicore::Tensor> InfinilmModel::default_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (nullptr == cache_config) {
        return {};
    }
    if (nullptr == text_config) {
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: text_config is null");
    }
    size_t head_dim = text_config->get<size_t>("head_dim");
    size_t num_key_value_heads = text_config->get<size_t>("num_key_value_heads");
    size_t max_position_embeddings = text_config->get<size_t>("max_position_embeddings");
    const auto &dtype = model_config_->get_kv_cache_dtype();
    if (dtype == infinicore::DataType::F8 && attention_backend != backends::AttentionBackend::PAGED_ATTN) {
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: FP8 KV cache is only supported on the PAGED_ATTN backend");
    }
    const size_t num_hidden_layers = text_config->get<size_t>("num_hidden_layers");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t pp_size = static_cast<size_t>(rank_info.pp_size);
    const size_t pp_stage = static_cast<size_t>(rank_info.pp_stage);
    const size_t local_layer_begin = num_hidden_layers * pp_stage / pp_size;
    const size_t local_layer_end = num_hidden_layers * (pp_stage + 1) / pp_size;

    std::vector<infinicore::Tensor> kv_cache_vec;
    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: invalid static kv cache config type");
        }
        kv_cache_vec.resize(num_hidden_layers);

        for (size_t layer_idx = local_layer_begin; layer_idx < local_layer_end; ++layer_idx) {
            auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
                head_dim,
                head_dim,
                num_key_value_heads,
                num_key_value_heads,
                max_position_embeddings,
                dtype,
                *static_kv_cache_config);
            kv_cache_vec[layer_idx] = kv_cache;
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN: {
        ;
    }
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error(
                "infinilm::InfinilmModel::default_allocate_kv_cache_tensors: invalid paged kv cache config type");
        }
        kv_cache_vec.resize(num_hidden_layers);

        // FP8(E4M3) KV cache: allocate per-layer k_scale/v_scale alongside the cache.
        // They live in ForwardContext::kv_scale_vec, parallel to kv_cache_vec, and are
        // consumed by PagedAttentionImpl. Cleared here so a non-FP8 reset drops stale scales.
        const bool kv_fp8 = (dtype == infinicore::DataType::F8);
        auto &kv_scale_vec = global_state::get_forward_context().kv_scale_vec;
        kv_scale_vec.clear();
        if (kv_fp8) {
            kv_scale_vec.resize(num_hidden_layers);
        }

        for (size_t layer_idx = local_layer_begin; layer_idx < local_layer_end; ++layer_idx) {
            auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
                head_dim,
                head_dim,
                num_key_value_heads,
                num_key_value_heads,
                dtype,
                *paged_kv_cache_config);
            kv_cache_vec[layer_idx] = kv_cache;
            if (kv_fp8) {
                kv_scale_vec[layer_idx] = cache::PagedKVCache::create_layer_kv_scales(
                    num_key_value_heads,
                    *paged_kv_cache_config);
            }
        }
        infinicore::context::syncStream();
        break;
    }
    default:
        throw std::runtime_error("infinilm::InfinilmModel::default_allocate_kv_cache_tensors: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
    return kv_cache_vec;
}

void InfinilmModel::process_weights_after_loading() {
    for (const auto &[_, sub] : children()) {
        process_weights_recursive_(sub.get());
    }
}

void InfinilmModel::reset_runtime_state() const {
    for (const auto &[_, sub] : children()) {
        reset_runtime_state_recursive_(sub.get());
    }
}

void InfinilmModel::process_weights_recursive_(infinicore::nn::Module *module) {
    for (const auto &[_, sub] : module->children()) {
        process_weights_recursive_(sub.get());
    }
    module->process_weights_after_loading();
}

void InfinilmModel::reset_runtime_state_recursive_(const infinicore::nn::Module *module) {
    for (const auto &[_, sub] : module->children()) {
        reset_runtime_state_recursive_(sub.get());
    }
    module->reset_runtime_state();
}

} // namespace infinilm
