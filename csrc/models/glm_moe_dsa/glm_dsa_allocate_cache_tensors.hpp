#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/cache.hpp"
#include "../../config/model_config.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::glm_moe_dsa {

struct GlmDsaCacheTensors {
    std::vector<infinicore::Tensor> mla;
    std::vector<infinicore::Tensor> mla_vendor;
    std::vector<infinicore::Tensor> indexer;
};

GlmDsaCacheTensors glm_dsa_allocate_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    const backends::AttentionBackend &attention_backend,
    size_t layer_start,
    size_t layer_end);

} // namespace infinilm::models::glm_moe_dsa
