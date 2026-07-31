#include "qwen3_5_moe_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include "../qwen3_5/qwen3_5_for_causal_lm.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_5_moe {

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_moe_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_5_moe" != model_type) {
        throw std::runtime_error(
            "create_qwen3_5_moe_model_config: model_type is not qwen3_5_moe");
    }

    model_config = qwen3_5::prepare_qwen3_5_model_config(model_config);
    auto &config_json = model_config->get_config_json();
    if (!config_json.contains("norm_topk_prob")) {
        config_json["norm_topk_prob"] = true;
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_5_moe

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_5_moe,
    infinilm::models::qwen3_5_moe::Qwen35MoeForConditionalGeneration,
    infinilm::models::qwen3_5_moe::create_qwen3_5_moe_model_config);
} // namespace
