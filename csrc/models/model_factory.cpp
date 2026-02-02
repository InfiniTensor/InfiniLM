#include "model_factory.hpp"
#include "llama/llama.hpp"

namespace infinilm {
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
std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    const InfinilmModel::Config &config,
    engine::distributed::RankInfo rank_info,
    const cache::CacheConfig *cache) {
    std::shared_ptr<InfinilmModel> model;
    if (const auto llama_config_ptr = dynamic_cast<const models::llama::LlamaConfig *>(&config)) {
        const auto &llama_config = *llama_config_ptr;
        model = std::make_shared<models::llama::LlamaForCausalLM>(
            llama_config, rank_info.device, rank_info);
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }

    if (cache) {
        model->reset_cache(cache);
    }

    return model;
}

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    engine::distributed::RankInfo rank_info,
    const cache::CacheConfig *cache) {

    std::shared_ptr<InfinilmModel> model;
    if (true) {
        model = std::make_shared<models::llama::LlamaForCausalLM>(
            model_config, rank_info.device, rank_info);
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }

    if (cache) {
        model->reset_cache(cache);
    }

    return model;
}
} // namespace infinilm
