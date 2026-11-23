#include "deepseek_v4_cache.hpp"

#include "../../parallel/topology.hpp"
#include "runtime/runtime.hpp"

#include <exception>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

infinilm::runtime::Tensor empty_zeroed(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Shape &shape,
    infinilm::runtime::DataType dtype) {
    auto tensor = infinilm::runtime::Tensor::empty(runtime, shape, dtype);
    if (tensor->nbytes() != 0) {
        runtime.setDeviceMemoryAsync(
            tensor->data(), 0, tensor->nbytes(), runtime.stream());
    }
    return tensor;
}

size_t layer_compress_ratio(
    const std::shared_ptr<infinilm::config::ModelConfig> &config,
    size_t layer_idx,
    size_t decoder_layers) {
    if (layer_idx >= decoder_layers) {
        return 1;
    }
    const auto &json = config->get_config_json();
    if (!json.contains("compress_ratios")
        || layer_idx >= json["compress_ratios"].size()) {
        return 0;
    }
    return json["compress_ratios"][layer_idx].get<size_t>();
}

std::vector<infinilm::runtime::Tensor> allocate_chunked_layer_views(
    infinilm::runtime::Runtime &runtime,
    size_t layer_count,
    size_t layers_per_allocation,
    const infinilm::runtime::Shape &layer_shape,
    infinilm::runtime::DataType dtype) {
    std::vector<infinilm::runtime::Tensor> views;
    views.reserve(layer_count);
    size_t cursor = 0;
    while (cursor < layer_count) {
        const size_t group = std::min(
            layers_per_allocation, layer_count - cursor);
        infinilm::runtime::Shape storage_shape;
        storage_shape.reserve(group + layer_shape.size());
        storage_shape.push_back(group);
        storage_shape.insert(
            storage_shape.end(), layer_shape.begin(), layer_shape.end());
        auto storage = empty_zeroed(runtime, storage_shape, dtype);
        for (size_t row = 0; row < group; ++row) {
            views.push_back(
                storage->narrow({{0, row, 1}})->squeeze(0));
        }
        cursor += group;
    }
    return views;
}

} // namespace

std::vector<infinilm::runtime::Tensor> allocate_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    backends::AttentionBackend attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    if (!model_config) {
        throw std::invalid_argument(
            "DeepseekV4 cache allocation requires model config");
    }
    if (attention_backend != backends::AttentionBackend::PAGED_ATTN
        && attention_backend != backends::AttentionBackend::FLASH_ATTN) {
        throw std::invalid_argument(
            "DeepseekV4 requires paged or flash attention cache");
    }
    cache::validate_backend_cache_config(cache_config, attention_backend);
    cache::validate_cache_runtime(runtime, rank);
    const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
    if (paged == nullptr) {
        throw std::invalid_argument("DeepseekV4 requires PagedKVCacheConfig");
    }

    const size_t decoder_layers = model_config->get<size_t>("num_hidden_layers");
    const size_t mtp_layers = model_config->get_or<size_t>("num_nextn_predict_layers", 0);
    const size_t total_layers = decoder_layers + mtp_layers;
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t q_lora_rank = model_config->get<size_t>("q_lora_rank");
    const size_t index_head_dim = model_config->get_or<size_t>(
        "index_head_dim", head_dim / 4);
    const size_t num_blocks = paged->num_blocks();
    const size_t block_size = paged->block_size();
    const auto state_block_size = [](const char *name, size_t fallback) {
        const char *raw = std::getenv(name);
        if (raw == nullptr) return fallback;
        const size_t value = static_cast<size_t>(std::stoull(raw));
        if (value == 0) throw std::invalid_argument(std::string(name) + " must be positive");
        return value;
    };
    const size_t state4_block_size = state_block_size(
        "INFINILM_DSV4_STATE4_BLOCK_SIZE", 4);
    const size_t state128_block_size = state_block_size(
        "INFINILM_DSV4_STATE128_BLOCK_SIZE", 8);
    if (block_size % state4_block_size != 0 || block_size % state128_block_size != 0) {
        throw std::invalid_argument("DeepSeek V4 state block sizes must divide block_size");
    }
    const auto dtype = model_config->get_kv_cache_dtype();
    const parallel::Topology topology(
        static_cast<size_t>(rank.tp_size), rank.pp_size, rank.pp_stage);
    const auto [local_begin, local_end] = topology.layer_range(total_layers);

    std::vector<size_t> compressed_layers;
    std::vector<size_t> ratio4_layers;
    std::vector<size_t> ratio128_layers;
    for (size_t layer = local_begin; layer < local_end; ++layer) {
        const size_t ratio = layer_compress_ratio(
            model_config, layer, decoder_layers);
        if (ratio > 1) {
            compressed_layers.push_back(layer);
            if (ratio == 4) {
                ratio4_layers.push_back(layer);
            } else if (ratio == 128) {
                ratio128_layers.push_back(layer);
            } else {
                throw std::invalid_argument(
                    "DeepseekV4 supports compressor ratios 4 and 128 only");
            }
        }
    }

    const size_t local_layer_count = local_end - local_begin;
    auto sliding_views = allocate_chunked_layer_views(
        runtime, local_layer_count, 48,
        {2, num_blocks, block_size, 1, head_dim}, dtype);
    auto compressed_views = allocate_chunked_layer_views(
        runtime, compressed_layers.size(), 48,
        {2, num_blocks, block_size, 1, head_dim}, dtype);
    auto ratio4_state_views = allocate_chunked_layer_views(
        runtime, ratio4_layers.size(), 7,
        {num_blocks * (block_size / state4_block_size), state4_block_size, 2 * q_lora_rank},
        infinilm::runtime::DataType::kFloat32);
    auto ratio128_state_views = allocate_chunked_layer_views(
        runtime, ratio128_layers.size(), 14,
        {num_blocks * (block_size / state128_block_size), state128_block_size, q_lora_rank},
        infinilm::runtime::DataType::kFloat32);
    auto indexer_state_views = allocate_chunked_layer_views(
        runtime, ratio4_layers.size(), 28,
        {num_blocks * (block_size / state4_block_size), state4_block_size, 4 * index_head_dim},
        infinilm::runtime::DataType::kFloat32);
    auto indexer_key_views = allocate_chunked_layer_views(
        runtime, ratio4_layers.size(), 56,
        {num_blocks, block_size, 1, index_head_dim},
        infinilm::runtime::DataType::kInt8);
    auto indexer_scale_views = allocate_chunked_layer_views(
        runtime, ratio4_layers.size(), 20,
        {num_blocks, block_size, 1},
        infinilm::runtime::DataType::kFloat16);

    std::vector<infinilm::runtime::Tensor> caches(
        total_layers * kCacheSlotsPerLayer);
    size_t compressed_cursor = 0;
    size_t ratio4_cursor = 0;
    size_t ratio128_cursor = 0;
    for (size_t local_row = 0; local_row < local_layer_count; ++local_row) {
        const size_t layer = local_begin + local_row;
        caches[cache_index(layer, CacheSlot::SlidingWindow)] =
            sliding_views[local_row];
        const size_t ratio = layer_compress_ratio(
            model_config, layer, decoder_layers);
        if (ratio <= 1) {
            continue;
        }
        caches[cache_index(layer, CacheSlot::CompressedKv)] =
            compressed_views[compressed_cursor++];
        if (ratio == 4) {
            caches[cache_index(layer, CacheSlot::CompressorState)] =
                ratio4_state_views[ratio4_cursor];
            caches[cache_index(layer, CacheSlot::IndexerState)] =
                indexer_state_views[ratio4_cursor];
            caches[cache_index(layer, CacheSlot::IndexerKey)] =
                indexer_key_views[ratio4_cursor];
            caches[cache_index(layer, CacheSlot::IndexerScale)] =
                indexer_scale_views[ratio4_cursor];
            ++ratio4_cursor;
        } else {
            caches[cache_index(layer, CacheSlot::CompressorState)] =
                ratio128_state_views[ratio128_cursor++];
        }
    }
    runtime.syncStream();
    return caches;
}

} // namespace infinilm::models::deepseek_v4
