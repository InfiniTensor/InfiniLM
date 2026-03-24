#include "models_registry.hpp"
#include "llama/llama_for_causal_lm.hpp"
#include "minicpm_sala/minicpm_sala_for_causal_lm.hpp"
#include "qwen3/qwen3_for_causal_lm.hpp"
#include "qwen3_moe/qwen3_moe_for_causal_lm.hpp"
#include "qwen3_next/qwen3_next_for_causal_lm.hpp"
#include "qwen3_vl/qwen3_vl_for_conditional_generation.hpp"

namespace infinilm::models {

void register_causal_lm_models(std::map<std::string, ModelCreator> &map) {
#define REGISTER_CAUSAL_LM_MODEL(model_type, model)                         \
    map[model_type] = [](std::shared_ptr<config::ModelConfig> config,       \
                         const infinicore::Device &device,                  \
                         engine::distributed::RankInfo rank_info,           \
                         backends::AttentionBackend backend) {              \
        return std::make_shared<model>(config, device, rank_info, backend); \
    }

    REGISTER_CAUSAL_LM_MODEL("llama", llama::LlamaForCausalLM);
    REGISTER_CAUSAL_LM_MODEL("fm9g", llama::LlamaForCausalLM);
    REGISTER_CAUSAL_LM_MODEL("qwen2", llama::LlamaForCausalLM);

    // REGISTER_CAUSAL_LM_MODEL("fm9g", fm9g::FM9GForCausalLM);
    // REGISTER_CAUSAL_LM_MODEL("qwen2", qwen2::Qwen2ForCausalLM);
    REGISTER_CAUSAL_LM_MODEL("qwen3", qwen3::Qwen3ForCausalLM);
    REGISTER_CAUSAL_LM_MODEL("qwen3_moe", qwen3_moe::Qwen3MoeForCausalLM);
    REGISTER_CAUSAL_LM_MODEL("qwen3_vl", qwen3_vl::Qwen3VLForConditionalGeneration);
    REGISTER_CAUSAL_LM_MODEL("qwen3_next", qwen3_next::Qwen3NextForCausalLM);
    REGISTER_CAUSAL_LM_MODEL("minicpm_sala", minicpm_sala::MiniCPMSALAForCausalLM);

#undef REGISTER_CAUSAL_LM_MODEL
}

void register_model_configs(std::map<std::string, ConfigCreator> &map) {
#define REGISTER_MODEL_CONFIG(model_type, creator) \
    map[model_type] = ConfigCreator(creator)

    REGISTER_MODEL_CONFIG("llama", llama::create_llama_model_config);
    REGISTER_MODEL_CONFIG("fm9g", llama::create_llama_model_config);
    REGISTER_MODEL_CONFIG("qwen2", llama::create_llama_model_config);

    // REGISTER_MODEL_CONFIG("llama", fm9g::create_fm9g_model_config); // llama -> fm9g
    // REGISTER_MODEL_CONFIG("fm9g", fm9g::create_fm9g_model_config);
    REGISTER_MODEL_CONFIG("qwen3", qwen3::create_qwen3_model_config);
    REGISTER_MODEL_CONFIG("qwen3_moe", qwen3_moe::create_qwen3_moe_model_config);
    REGISTER_MODEL_CONFIG("qwen3_vl", qwen3_vl::create_qwen3_vl_model_config);
    REGISTER_MODEL_CONFIG("qwen3_next", qwen3_next::create_qwen3_next_model_config);
    REGISTER_MODEL_CONFIG("minicpm_sala", minicpm_sala::create_minicpm_sala_model_config);

#undef REGISTER_MODEL_CONFIG
}

} // namespace infinilm::models
