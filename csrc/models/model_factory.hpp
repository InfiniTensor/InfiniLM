#pragma once

#include "../config/model_config.hpp"
#include "infinilm_model.hpp"

#include "../engine/distributed/distributed.hpp"

namespace infinilm {
class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr);
};
} // namespace infinilm
