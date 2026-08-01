#include "glm_dsa_allocate_cache_tensors.hpp"

#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <cstdlib>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::models::glm_moe_dsa {

namespace {
bool layer_uses_indexer_cache(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    size_t layer_idx) {
    const auto &config_json = model_config->get_config_json();
    const auto indexer_types = config_json.value(
        "indexer_types", nlohmann::json::array());
    if (layer_idx < indexer_types.size()) {
        return indexer_types[layer_idx].get<std::string>() != "shared";
    }
    return true;
}
} // namespace

GlmDsaCacheTensors glm_dsa_allocate_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend,
    size_t layer_start,
    size_t layer_end) {
    if (cache_config == nullptr) {
        return {};
    }
    if (model_config == nullptr) {
        throw std::runtime_error("glm_dsa_allocate_cache_tensors: model config is null");
    }
    if (attention_backend == backends::AttentionBackend::STATIC_ATTN) {
        throw std::runtime_error("GLM-5.2 DSA requires paged cache");
    }
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (paged_config == nullptr) {
        throw std::runtime_error("glm_dsa_allocate_cache_tensors: expected paged cache config");
    }
    if (paged_config->block_size() != 64) {
        throw std::runtime_error("GLM-5.2 DSA requires block_size=64");
    }

    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    if (layer_start > layer_end || layer_end > num_layers) {
        throw std::runtime_error(
            "glm_dsa_allocate_cache_tensors: invalid pipeline layer range");
    }
    const size_t kv_lora_rank = model_config->get<size_t>("kv_lora_rank");
    const size_t rope_dim = model_config->get<size_t>("qk_rope_head_dim");
    const bool use_vendor_shadow = std::getenv("INFINILM_GLM_FP8_SPARSE_VENDOR") != nullptr;
    const size_t vendor_cache_stride = kv_lora_rank + rope_dim;
    const size_t mla_cache_stride = kv_lora_rank + 4 * sizeof(float)
                                  + rope_dim * sizeof(uint16_t);
    const size_t index_dim = model_config->get<size_t>("index_head_dim");
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    const auto &device = rank_info.device;

    GlmDsaCacheTensors caches;
    caches.mla.reserve(num_layers);
    if (use_vendor_shadow) {
        caches.mla_vendor.reserve(num_layers);
    }
    caches.indexer.reserve(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        if (layer < layer_start || layer >= layer_end) {
            caches.mla.emplace_back();
            if (use_vendor_shadow) {
                caches.mla_vendor.emplace_back();
            }
            caches.indexer.emplace_back();
            continue;
        }
        auto mla = infinicore::Tensor::empty(
            {paged_config->num_blocks(), paged_config->block_size(), mla_cache_stride},
            infinicore::DataType::U8,
            device);
        set_zeros(mla);
        caches.mla.push_back(std::move(mla));

        if (use_vendor_shadow) {
            auto vendor_cache = infinicore::Tensor::empty(
                {paged_config->num_blocks(), paged_config->block_size(),
                 vendor_cache_stride},
                infinicore::DataType::BF16,
                device);
            set_zeros(vendor_cache);
            caches.mla_vendor.push_back(std::move(vendor_cache));
        }

        infinicore::Tensor indexer;
        if (layer_uses_indexer_cache(model_config, layer)) {
            indexer = infinicore::Tensor::empty(
                {paged_config->num_blocks(), paged_config->block_size(), index_dim + sizeof(float)},
                infinicore::DataType::U8, device);
            set_zeros(indexer);
        }
        caches.indexer.push_back(std::move(indexer));
    }
    if (rank_info.tp_rank == 0) {
        size_t indexer_layers = 0;
        for (size_t layer = layer_start; layer < layer_end; ++layer) {
            indexer_layers += layer_uses_indexer_cache(model_config, layer) ? 1 : 0;
        }
        spdlog::info(
            "GLM DSA paged cache: physical_blocks={}, kernel_block_size={}, "
            "mla_layout=fp8_ds_mla({}+4xfp32+{}xbf16={}B), "
            "index_dim={}+fp32_scale, layers=[{},{}), indexer_layers={}, "
            "indexer_cache_tp_ranks={}, vendor_shadow={}, "
            "vendor_token_bytes={}",
            paged_config->num_blocks(),
            paged_config->block_size(),
            kv_lora_rank, rope_dim, mla_cache_stride,
            index_dim,
            layer_start,
            layer_end,
            indexer_layers,
            rank_info.tp_size,
            use_vendor_shadow ? "enabled" : "disabled",
            vendor_cache_stride * sizeof(uint16_t));
    }
    infinicore::context::syncStream();
    return caches;
}

} // namespace infinilm::models::glm_moe_dsa
