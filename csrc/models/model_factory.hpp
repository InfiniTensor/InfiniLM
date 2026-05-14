#pragma once

#include "../backends/attention_backends.hpp"
#include "../engine/distributed/distributed.hpp"
#include "infinilm_model.hpp"

namespace infinilm {

class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr,
        backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device,
        const cache::CacheConfig *cache = nullptr);
};
} // namespace infinilm
