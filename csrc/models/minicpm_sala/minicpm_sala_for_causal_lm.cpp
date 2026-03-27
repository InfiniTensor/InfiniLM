#include "minicpm_sala_for_causal_lm.hpp"
#include "../../config/infinilm_config.hpp"
#include "../../engine/forward_context.hpp"
#include "../models_registry.hpp"
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm_sala {

MiniCPMSALAForCausalLM::MiniCPMSALAForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device) {
    model_config_ = model_config;
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output MiniCPMSALAForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void MiniCPMSALAForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto &kv_cache_vec = engine::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::config::get_current_infinilm_config().attention_backend;

    auto new_kv_cache_vec = minicpm_sala_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend);
    kv_cache_vec = std::move(new_kv_cache_vec);
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_sala_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpm_sala" != model_type) {
        throw std::runtime_error("infinilm::models::minicpm_sala::create_minicpm_sala_model_config: model_type is not minicpm_sala");
    }
    return model_config;
}

} // namespace infinilm::models::minicpm_sala

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm_sala,
    infinilm::models::minicpm_sala::MiniCPMSALAForCausalLM,
    infinilm::models::minicpm_sala::create_minicpm_sala_model_config);
} // namespace
