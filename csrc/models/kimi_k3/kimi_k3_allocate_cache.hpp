#pragma once

#include "../qwen3_next/qwen3_next_allocate_kv_cache_tensors.hpp"

#include <memory>

namespace infinilm::models::kimi_k3 {

qwen3_next::AllocatedHybridCache kimi_k3_allocate_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    backends::AttentionBackend attention_backend);

} // namespace infinilm::models::kimi_k3
