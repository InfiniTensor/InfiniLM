#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::gemma4 {

std::vector<infinilm::runtime::Tensor> gemma4_allocate_kv_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank_info,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend);

} // namespace infinilm::models::gemma4
