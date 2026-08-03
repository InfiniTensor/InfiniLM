#include "qwen3_next_for_causal_lm.hpp"
#include "../../cache/hybrid_cache.hpp"
#include "../../config/hybrid_model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::qwen3_next {

Qwen3NextForCausalLM::Qwen3NextForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device) {
    model_config_ = model_config;
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output Qwen3NextForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void Qwen3NextForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.clear_model_caches();

    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;

    auto cache_vectors = cache::allocate_hybrid_cache_tensors(cache_config, model_config_, attention_backend);
    forward_context.kv_cache_vec = std::move(cache_vectors.kv_cache_tensors);
    forward_context.conv_state_vec = std::move(cache_vectors.conv_state_tensors);
    forward_context.ssm_state_vec = std::move(cache_vectors.ssm_state_tensors);
    forward_context.mamba_state_pool_size = cache_vectors.mamba_state_pool_size;
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_next" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_next::create_qwen3_next_model_config: model_type is not qwen3_next");
    }

    infinilm::config::prepare_hybrid_model_config(model_config);
    return model_config;
}

} // namespace infinilm::models::qwen3_next

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_next,
    infinilm::models::qwen3_next::Qwen3NextForCausalLM,
    infinilm::models::qwen3_next::create_qwen3_next_model_config);
} // namespace
