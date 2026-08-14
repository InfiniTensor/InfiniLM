#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../cache/mamba_cache.hpp"
#include "../../config/model_config.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_next {

struct AllocatedHybridCache {
    std::vector<infinicore::Tensor> kv_cache_tensors;
    std::vector<infinicore::Tensor> conv_state_tensors;
    std::vector<infinicore::Tensor> ssm_state_tensors;
};

AllocatedHybridCache qwen3_next_allocate_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend);

} // namespace infinilm::models::qwen3_next
