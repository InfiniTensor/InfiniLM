#include "deepseek_decoder_layer.hpp"
#include <exception>
#include <stdexcept>
#include "runtime/runtime.hpp"


namespace infinilm::models::deepseek {

DeepseekDecoderLayer::DeepseekDecoderLayer(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("DeepseekDecoderLayer construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Deepseek construction resources");
    try {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        input_layernorm_ = this->bind_child("input_layernorm", std::make_shared<std::remove_reference_t<decltype(*input_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        post_attention_layernorm_ = this->bind_child("post_attention_layernorm", std::make_shared<std::remove_reference_t<decltype(*post_attention_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        self_attn_ = this->bind_child("self_attn", std::make_shared<std::remove_reference_t<decltype(*self_attn_ )>>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));

    const size_t first_k_dense_replace = model_config->get_or<size_t>("first_k_dense_replace", 0);
    const size_t moe_layer_freq = model_config->get_or<size_t>("moe_layer_freq", 1);
    const bool use_moe = model_config->get_or<size_t>("n_routed_experts", 0) > 0
                      && layer_idx >= first_k_dense_replace
                      && (moe_layer_freq == 0 || layer_idx % moe_layer_freq == 0);

    if (use_moe) {
        mlp_ = std::make_shared<DeepseekMLP>(
                this->bind_child("mlp", std::make_shared<deepseek_v2::DeepseekV2MoE>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config)));
    } else {
        mlp_ = std::make_shared<DeepseekMLP>(
                this->bind_child("mlp", std::make_shared<deepseek_v2::DeepseekV2MLP>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config)));
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor>
DeepseekDecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                              infinilm::runtime::Tensor &hidden_states,
                              infinilm::runtime::Tensor &residual,
                              infinilm::global_state::ForwardContext &forward_context) {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    input_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states, forward_context);
    post_attention_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = std::visit(
        [&](auto &mlp_ptr) { return mlp_ptr->forward(runtime, hidden_states); }, *mlp_);
    return std::make_tuple(hidden_states, residual);
}


} // namespace infinilm::models::deepseek
