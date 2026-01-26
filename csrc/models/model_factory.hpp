#pragma once

#include "../config/global_config.hpp"
#include "infinilm_model.hpp"

#include "../engine/distributed/distributed.hpp"

namespace infinilm {
class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::global_config::GlobalConfig> global_config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr);
};
} // namespace infinilm
