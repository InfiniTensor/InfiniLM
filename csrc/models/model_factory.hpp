#pragma once

#include "../config/model_config.hpp"
#include "infinilm_model.hpp"

#include "../engine/distributed/distributed.hpp"

namespace infinilm {
class InfinilmModelFactory {
public:
    /**
     * @deprecated This function is deprecated and will be REMOVED in the next major release (v0.2.0).
     *
     * ⚠️ DEVELOPMENT POLICY:
     *   - NO new development or feature additions permitted on this interface
     *   - Only critical bug fixes (security/stability) allowed until removal
     *   - All new code MUST migrate to the polymorphic overload below
     *
     * Replacement: Use the polymorphic overload of this same function name with updated signature
     * Reason: Legacy signature lacks support for dynamic quantization modes.
     * Removal target: v0.2.0 (Q2 2026)
     */
    static std::shared_ptr<InfinilmModel> createModel(
        const InfinilmModel::Config &config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr);

    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
        const cache::CacheConfig *cache = nullptr);
};
} // namespace infinilm
