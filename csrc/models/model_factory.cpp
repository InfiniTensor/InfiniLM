#include "model_factory.hpp"
#include "llama/llama.hpp"

namespace infinilm {
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
} // namespace infinilm
