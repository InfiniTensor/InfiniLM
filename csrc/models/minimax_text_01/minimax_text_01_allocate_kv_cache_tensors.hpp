#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../cache/mamba_cache.hpp"
#include "../../config/model_config.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::minimax_text_01 {

struct AllocatedHybridCache {
    std::vector<infinicore::Tensor> kv_cache_tensors;
    std::vector<infinicore::Tensor> ssm_state_tensors;
};

AllocatedHybridCache minimax_text_01_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend);

} // namespace infinilm::models::minimax_text_01
