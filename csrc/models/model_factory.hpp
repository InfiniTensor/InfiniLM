#pragma once

#include "infinilm_model.hpp"

#include "../engine/distributed/distributed.hpp"

namespace infinilm {
class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        const InfinilmModel::Config &config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr);
};
} // namespace infinilm
