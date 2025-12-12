#pragma once

#include "infinilm_model.hpp"

#include "../engine/distributed/distributed.hpp"

namespace infinilm {
class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(const std::any &config, engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(), std::shared_ptr<cache::CacheInterface> cache_ptr = nullptr);
};
} // namespace infinilm
