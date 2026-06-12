#include "model_factory.hpp"
#include "models_registry.hpp"

namespace infinilm {

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device,
    const cache::CacheConfig *cache) {
    const std::string model_type = model_config->get<std::string>("model_type");
    std::shared_ptr<InfinilmModel> model;
    const auto &model_map = models::get_causal_lm_model_map();
    auto it = model_map.find(model_type);
    if (it != model_map.end()) {
        // create model
        auto &model_creator = it->second;
        model = model_creator(model_config, device);
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model_type");
    }

    if (cache) {
        model->reset_cache(cache);
    }
    return model;
}
} // namespace infinilm
