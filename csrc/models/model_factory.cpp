#include "model_factory.hpp"
// #include "fm9g/fm9g_for_causal_lm.hpp"
#include "llama/llama_for_causal_lm.hpp"
#include "minicpm_sala/minicpm_sala_for_causal_lm.hpp"
#include "qwen2/qwen2_for_causal_lm.hpp"
#include "qwen3/qwen3_for_causal_lm.hpp"
#include "qwen3_moe/qwen3_moe_for_causal_lm.hpp"
// #include "qwen3_next/qwen3_next_for_causal_lm.hpp"

namespace infinilm {

std::map<std::string, ModelCreator> &InfinilmModelFactory::_modelsForCausalLM() {
    static std::map<std::string, ModelCreator> _map;

#define REGISTER_CAUSAL_LM_MODEL(model_type, model)                         \
    _map[model_type] = [](std::shared_ptr<config::ModelConfig> config,      \
                          const infinicore::Device &device,                 \
                          engine::distributed::RankInfo rank_info,          \
                          backends::AttentionBackend backend) {             \
        return std::make_shared<model>(config, device, rank_info, backend); \
    }

    if (_map.empty()) {
        // REGISTER_CAUSAL_LM_MODEL("llama", models::fm9g::FM9GForCausalLM); // llama -> fm9g
        // REGISTER_CAUSAL_LM_MODEL("fm9g", models::fm9g::FM9GForCausalLM);
        REGISTER_CAUSAL_LM_MODEL("qwen2", models::qwen2::Qwen2ForCausalLM);
        REGISTER_CAUSAL_LM_MODEL("qwen3", models::qwen3::Qwen3ForCausalLM);
        REGISTER_CAUSAL_LM_MODEL("qwen3_moe", models::qwen3_moe::Qwen3MoeForCausalLM);
        REGISTER_CAUSAL_LM_MODEL("minicpm_sala", models::minicpm_sala::MiniCPMSALAForCausalLM);
        // REGISTER_CAUSAL_LM_MODEL("qwen3_next", models::qwen3_next::Qwen3NextForCausalLM);
    }
#undef REGISTER_CAUSAL_LM_MODEL

    return _map;
}

std::shared_ptr<InfinilmModel> InfinilmModelFactory::createModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    engine::distributed::RankInfo rank_info,
    const cache::CacheConfig *cache,
    backends::AttentionBackend attention_backend) {

    const std::string model_type = model_config->get<std::string>("model_type");

    std::shared_ptr<InfinilmModel> model;
    auto it = _modelsForCausalLM().find(model_type);
    if (it != _modelsForCausalLM().end()) {
        ModelCreator model_creator = it->second;
        model = model_creator(model_config, rank_info.device, rank_info, attention_backend);
    } else {
        throw std::invalid_argument("InfinilmModelFactory::createModel: Unsupported model config type");
    }

    if (cache) {
        model->reset_cache(cache);
    }

    return model;
}

} // namespace infinilm
