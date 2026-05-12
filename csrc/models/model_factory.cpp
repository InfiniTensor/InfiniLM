#include "model_factory.hpp"
#include "llama_legacy/llama_for_causal_lm.hpp"
#include "models_registry.hpp"

namespace infinilm {

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    engine::distributed::RankInfo rank_info,
    const cache::CacheConfig *cache,
    backends::AttentionBackend attention_backend) {
    std::shared_ptr<InfinilmModel> model;
    if (true) {
        model = std::make_shared<models::llama_legacy::LlamaForCausalLM>(
            model_config, rank_info.device, rank_info, attention_backend);
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
