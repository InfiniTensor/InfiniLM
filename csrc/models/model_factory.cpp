#include "model_factory.hpp"
#include "llama/llama.hpp"

namespace infinilm {
std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    const InfinilmModel::Config &config,
    engine::distributed::RankInfo rank_info,
    std::shared_ptr<cache::DynamicCache> cache_ptr) {

    if (const auto llama_config_ptr = dynamic_cast<const models::llama::LlamaConfig *>(&config)) {
        const auto &llama_config = *llama_config_ptr;
        auto model = std::make_shared<models::llama::LlamaForCausalLM>(
            llama_config, rank_info.device, rank_info);

        if (cache_ptr != nullptr) {
            model->model().set_external_cache(cache_ptr);
        }

        return model;
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }
}
} // namespace infinilm
