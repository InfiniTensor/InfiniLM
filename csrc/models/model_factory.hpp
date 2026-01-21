#pragma once

#include "../config/global_config.hpp"
#include "infinilm_model.hpp"

#include "../engine/distributed/distributed.hpp"

namespace infinilm {
class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        const InfinilmModel::Config &config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr,
        std::shared_ptr<infinilm::config::global_config::GlobalConfig> global_config = nullptr);
};
} // namespace infinilm
