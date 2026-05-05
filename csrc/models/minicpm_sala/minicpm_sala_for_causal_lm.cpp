#include "minicpm_sala_for_causal_lm.hpp"
#include "../models_registry.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm_sala {

std::vector<infinicore::Tensor> minicpm_sala_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend);

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_sala_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpm_sala" != model_type) {
        throw std::runtime_error("infinilm::models::minicpm_sala::create_minicpm_sala_model_config: model_type is not minicpm_sala");
    }
    return model_config;
}

MiniCPMSALAForCausalLM::MiniCPMSALAForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    device_ = device;
    model_config_ = model_config;

    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();
    INFINICORE_NN_MODULE_INIT(model, model_config, device);

    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

MiniCPMSALAForCausalLM::Output MiniCPMSALAForCausalLM::forward(
    const Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();

    auto hidden_states = model_->forward(input_ids, position_ids);

    // MuP lm_head scale baked into lm_head.weight at load time; no forward scaling here.
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void MiniCPMSALAForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    // Match `InfinilmModel::reset_cache`: own `cache_config_` + `kv_cache_vec` here; inner model only
    // resets per-layer attention state. MiniCPM uses `minicpm_sala_allocate_kv_cache_tensors` instead of
    // `default_allocate_kv_cache_tensors`.
    if (cache_config == nullptr) {
        cache_config_.reset();
        infinilm::global_state::get_forward_context().kv_cache_vec.clear();
        model_->reset_state();
        return;
    }
    cache_config_ = cache_config->unique_copy();
    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend =
        infinilm::global_state::get_infinilm_config().attention_backend;
    kv_cache_vec = std::move(
        minicpm_sala_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend));
    model_->reset_state();
}

} // namespace infinilm::models::minicpm_sala

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpm_sala,
    infinilm::models::minicpm_sala::MiniCPMSALAForCausalLM,
    infinilm::models::minicpm_sala::create_minicpm_sala_model_config);
} // namespace

