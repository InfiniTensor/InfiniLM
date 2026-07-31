#include "qwen3_5_moe_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include "../qwen3_5/qwen3_5_for_causal_lm.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::qwen3_5_moe {

Qwen35MoeForConditionalGeneration::Qwen35MoeForConditionalGeneration(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

InfinilmModel::Output Qwen35MoeForConditionalGeneration::forward(
    const InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    return {lm_head_->forward(hidden_states)};
}

void Qwen35MoeForConditionalGeneration::reset_cache(
    const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
    } else {
        cache_config_ = cache_config->unique_copy();
    }
    model_->reset_cache(cache_config);
}

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
