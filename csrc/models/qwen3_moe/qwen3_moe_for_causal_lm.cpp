#include "qwen3_moe_for_causal_lm.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_moe {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_moe_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_moe" != model_type) {
        throw std::runtime_error("create_qwen3_moe_model_config: model_type is not qwen3_moe");
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_moe

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_moe,
    infinilm::models::qwen3_moe::Qwen3MoeForCausalLM,
    infinilm::models::qwen3_moe::create_qwen3_moe_model_config);
} // namespace
