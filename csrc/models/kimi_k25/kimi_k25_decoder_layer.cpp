#include "kimi_k25_decoder_layer.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::models::kimi_k25 {

KimiK25DecoderLayer::KimiK25DecoderLayer(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("KimiK25DecoderLayer construction requires eager Runtime");
    }
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Invalid explicit KimiK25DecoderLayer construction resources");
    }
    try {
        const auto &dtype = model_config->get_dtype();
        const size_t hidden_size = model_config->get<size_t>("hidden_size");
        const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        self_attn_ = std::make_shared<std::remove_reference_t<decltype(*self_attn_)>>(
            runtime, model_config, layer_idx, rank_info, attention_backend);
        this->bind_child("self_attn", self_attn_);
        input_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*input_layernorm_)>>(
            runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("input_layernorm", input_layernorm_);
        post_attention_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*post_attention_layernorm_)>>(
            runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("post_attention_layernorm", post_attention_layernorm_);

        const size_t first_dense_layers = model_config->get_or<size_t>("first_k_dense_replace", 0);
        const size_t moe_layer_freq = model_config->get_or<size_t>("moe_layer_freq", 1);
        use_moe_ = model_config->get_or<size_t>("n_routed_experts", 0) > 0
                && layer_idx >= first_dense_layers
                && (moe_layer_freq == 0 || layer_idx % moe_layer_freq == 0);
        if (use_moe_) {
            moe_mlp_ = std::make_shared<KimiK25MoE>(
                runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("mlp", moe_mlp_);
        } else {
            auto dense_config = make_kimi_mlp_config(
                model_config, model_config->get<size_t>("intermediate_size"));
            dense_mlp_ = std::make_shared<infinilm::layers::mlp::MLP>(
                runtime, dense_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("mlp", dense_mlp_);
        }
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor>
KimiK25DecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                             infinilm::runtime::Tensor &hidden_states,
                             infinilm::runtime::Tensor &residual,
                             global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    input_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states, forward_context);
    post_attention_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = use_moe_ ? moe_mlp_->forward(runtime, hidden_states) : dense_mlp_->forward(runtime, hidden_states);
    return {hidden_states, residual};
}

} // namespace infinilm::models::kimi_k25
