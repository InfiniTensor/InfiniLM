#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/dist_config.hpp"
#include "runtime/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::runtime { class Runtime; }

namespace infinilm::models::deepseek_v4 {

enum class CacheSlot : size_t {
    SlidingWindow = 0,
    CompressedKv = 1,
    CompressorState = 2,
    IndexerState = 3,
    IndexerKey = 4,
    IndexerScale = 5,
};

constexpr size_t kCacheSlotsPerLayer = 6;

constexpr size_t cache_index(size_t layer_idx, CacheSlot slot) {
    return layer_idx * kCacheSlotsPerLayer + static_cast<size_t>(slot);
}

std::vector<infinilm::runtime::Tensor> allocate_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    backends::AttentionBackend attention_backend);

} // namespace infinilm::models::deepseek_v4
