#include "deepseek_v2_decoder_layer.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>


namespace infinilm::models::deepseek_v2 {

DeepseekV2DecoderLayer::DeepseekV2DecoderLayer(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config,
    bool use_mla) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("DeepseekV2DecoderLayer construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Deepseek construction resources");
    if (use_mla && model_config->get_or<size_t>("q_lora_rank", 0) != 0)
        throw std::runtime_error("DeepseekV2MLAAttention: q_lora_rank is not supported yet");
    try {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        input_layernorm_ = this->bind_child("input_layernorm", std::make_shared<std::remove_reference_t<decltype(*input_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        post_attention_layernorm_ = this->bind_child("post_attention_layernorm", std::make_shared<std::remove_reference_t<decltype(*post_attention_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        if (use_mla) {
            self_attn_ = std::make_shared<DeepseekV2SelfAttention>(this->bind_child("self_attn", std::make_shared<DeepseekV2MLAAttention>(runtime, model_config, layer_idx, rank_info, attention_backend)));
    } else {
            self_attn_ = std::make_shared<DeepseekV2SelfAttention>(this->bind_child("self_attn", std::make_shared<DeepseekV2Attention>(runtime, model_config, layer_idx, rank_info, attention_backend)));
    }

    const size_t first_k_dense_replace = model_config->get_or<size_t>("first_k_dense_replace", 0);
    const size_t moe_layer_freq = model_config->get_or<size_t>("moe_layer_freq", 1);
    use_moe_ = model_config->get_or<size_t>("n_routed_experts", 0) > 0
            && layer_idx >= first_k_dense_replace
            && (moe_layer_freq == 0 || layer_idx % moe_layer_freq == 0);
    if (use_moe_) {
            moe_mlp_ = this->bind_child("mlp", std::make_shared<DeepseekV2MoE>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));
    } else {
            dense_mlp_ = this->bind_child("mlp", std::make_shared<DeepseekV2MLP>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor>
DeepseekV2DecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                                infinilm::runtime::Tensor &hidden_states,
                                infinilm::runtime::Tensor &residual,
                                infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    input_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = std::visit(
        [&](auto &attn_ptr) { return attn_ptr->forward(positions, hidden_states, forward_context); }, *self_attn_);
    post_attention_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = use_moe_ ? moe_mlp_->forward(runtime, hidden_states) : dense_mlp_->forward(runtime, hidden_states);
    return {hidden_states, residual};
}

} // namespace infinilm::models::deepseek_v2
