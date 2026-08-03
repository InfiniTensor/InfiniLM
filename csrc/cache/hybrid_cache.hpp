#pragma once

#include "../backends/attention_backends.hpp"
#include "../config/model_config.hpp"
#include "kv_cache.hpp"
#include "mamba_cache.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::cache {

struct HybridCacheTensors {
    std::vector<infinicore::Tensor> kv_cache_tensors;
    std::vector<infinicore::Tensor> conv_state_tensors;
    std::vector<infinicore::Tensor> ssm_state_tensors;
    size_t mamba_state_pool_size{0};
};

HybridCacheTensors allocate_hybrid_cache_tensors(
    const CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend);

} // namespace infinilm::cache
