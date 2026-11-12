#include "zero21_for_causal_lm.hpp"

#include "../models_registry.hpp"

namespace infinilm::models::zero21 {

std::shared_ptr<infinilm::config::ModelConfig> create_zero21_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    return infinilm::models::deepseek_v2::create_deepseek_v2_style_model_config(
        std::move(model_config), "zero21_model");
}

} // namespace infinilm::models::zero21

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    zero21_model,
    infinilm::models::zero21::Zero21ModelForCausalLM,
    infinilm::models::zero21::create_zero21_model_config);
} // namespace
