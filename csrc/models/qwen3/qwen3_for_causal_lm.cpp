#include "qwen3_for_causal_lm.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3 {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("qwen3" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3::create_qwen3_model_config: model_type is not qwen3");
    }

    return model_config;
}

} // namespace infinilm::models::qwen3

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3,
    infinilm::models::qwen3::Qwen3ForCausalLM,
    infinilm::models::qwen3::create_qwen3_model_config);

} // namespace
