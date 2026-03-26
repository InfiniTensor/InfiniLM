#include "qwen2_for_causal_lm.hpp"
#include "../models_registry.hpp"

namespace infinilm::models::qwen2 {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen2" != model_type) {
        throw std::runtime_error("infinilm::models::qwen2::create_qwen2_model_config: model_type is not qwen2");
    }

    return model_config;
}

} // namespace infinilm::models::qwen2

namespace {

// INFINILM_REGISTER_CAUSAL_LM_MODEL(
//     qwen2,
//     infinilm::models::qwen2::Qwen2ForCausalLM,
//     infinilm::models::qwen2::create_qwen2_model_config);

} // namespace
