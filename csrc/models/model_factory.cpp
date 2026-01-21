#include "model_factory.hpp"
#include "llama/llama.hpp"
#include "llava/llava_model.hpp"
#include "minicpmv/minicpmv_model.hpp"

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
    } else if (const auto llava_config_ptr = dynamic_cast<const models::llava::LlavaConfig *>(&config)) {
        const auto &llava_config = *llava_config_ptr;
        model = std::make_shared<models::llava::LlavaForConditionalGeneration>(
            llava_config, rank_info.device, rank_info);
    } else if (const auto minicpmv_config_ptr = dynamic_cast<const models::minicpmv::MiniCPMVConfig *>(&config)) {
        const auto &minicpmv_config = *minicpmv_config_ptr;
        model = std::make_shared<models::minicpmv::MiniCPMVModel>(
            minicpmv_config, rank_info.device, rank_info);
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }

    if (cache) {
        model->reset_cache(cache);
    }

    return model;
}
} // namespace infinilm
