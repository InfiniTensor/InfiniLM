#pragma once

#include "infinilm_model.hpp"
#include "models_registry.hpp"

namespace infinilm {

class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<config::ModelConfig>, infinilm::runtime::Runtime &,
        const engine::distributed::RankInfo &, backends::AttentionBackend,
        const layers::moe::EPConfig &, const cache::CacheConfig *, global_state::ForwardContext &, bool use_mla);
};
} // namespace infinilm
