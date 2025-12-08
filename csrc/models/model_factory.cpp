#include "model_factory.hpp"
#include "llama/llama.hpp"

namespace infinilm {
std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(const std::any &config, engine::distributed::RankInfo rank_info) {

    if (config.type() == typeid(models::llama::LlamaConfig)) {
        const auto &llama_config = std::any_cast<models::llama::LlamaConfig>(config);
        return std::make_shared<models::llama::LlamaForCausalLM>(llama_config, rank_info.device, infinicore::DataType::BF16, rank_info);
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }
}
} // namespace infinilm
