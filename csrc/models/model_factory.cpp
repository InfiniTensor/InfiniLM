#include "model_factory.hpp"
#include "llama/llama_for_causal_lm.hpp"
#include "qwen3/qwen3_for_causal_lm.hpp"
#include "qwen3_moe/qwen3_moe_for_causal_lm.hpp"


namespace infinilm {

std::map<std::string, ModelCreator> &InfinilmModelFactory::_ModelsForCausalLM() {
    static std::map<std::string, ModelCreator> _map;

    #define REGISTER_CAUSAL_LM_MODEL(model_type, model)                            \
        _map[model_type] = [](std::shared_ptr<config::ModelConfig> config,                 \
                            const infinicore::Device &device,                            \
                            engine::distributed::RankInfo rank_info,                    \
                            backends::AttentionBackend backend) {                        \
            return std::make_shared<model>(config, device, rank_info, backend);       \
        }

    REGISTER_CAUSAL_LM_MODEL("qwen3", models::qwen3::Qwen3ForCausalLM);

    #undef REGISTER_CAUSAL_LM_MODEL

    return _map;
}


std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    engine::distributed::RankInfo rank_info,
    const cache::CacheConfig *cache,
    backends::AttentionBackend attention_backend) {

    std::shared_ptr<InfinilmModel> model;
    const std::string model_type = model_config->get<std::string>("model_type");

    std::cout << " ???????????? " << std::endl;
    for(auto &[model_type, model] : _ModelsForCausalLM()) {
        std::cout << "_ModelsForCausalLM:: model_type: " << model_type << std::endl;

    }

    // auto it = _ModelsForCausalLM().find(model_type);
    // if (it != _ModelsForCausalLM().end()) {
    //     model = it->second(model_config, rank_info.device, rank_info, attention_backend);
    // } 
    
    
    // if ("llama" == model_type or "fm9g" == model_type or "qwen2" == model_type) {
    //     model = std::make_shared<models::llama::LlamaForCausalLM>(
    //         model_config, rank_info.device, rank_info, attention_backend);
    // } else if ("qwen3_moe" == model_type) {
    //     model = std::make_shared<models::qwen3_moe::Qwen3MoeForCausalLM>(
    //         model_config, rank_info.device, rank_info, attention_backend);
    // } else {
    //     throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    // }

    if (cache) {
        model->reset_cache(cache);
    }

    return model;
}

}  // namespace infinilm
