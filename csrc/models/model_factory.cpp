#include "model_factory.hpp"
#include "llama/llama.hpp"

namespace infinilm {
std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    const std::any &config,
    engine::distributed::RankInfo rank_info,
    std::shared_ptr<cache::DynamicCache> cache_ptr) {

    if (config.type() == typeid(models::llama::LlamaConfig)) {
        const auto &llama_config = std::any_cast<models::llama::LlamaConfig>(config);
        auto model = std::make_shared<models::llama::LlamaForCausalLM>(
            llama_config, rank_info.device, infinicore::DataType::BF16, rank_info);

        if (cache_ptr != nullptr) {
            model->model().set_external_cache(cache_ptr);
        }

        return model;
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }
}
} // namespace infinilm
