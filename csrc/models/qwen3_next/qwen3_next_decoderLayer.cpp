#include "qwen3_next_decoderLayer.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextDecoderLayer::Qwen3NextDecoderLayer(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config)
    : layer_idx_(layer_idx) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || !model_config || rank_info.device != runtime.device()
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm)) {
        throw std::invalid_argument("Next construction requires matching eager Runtime/rank");
    }
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const auto layer_types = model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types.at(layer_idx);
    if (layer_type_ != "linear_attention" && layer_type_ != "full_attention") {
        throw std::invalid_argument("Qwen3NextDecoderLayer: unsupported layer_type");
    }
    try {
        input_layernorm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("input_layernorm", input_layernorm_);
        post_attention_layernorm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("post_attention_layernorm", post_attention_layernorm_);
        mlp_ = std::make_shared<Qwen3NextSparseMoeBlock>(
            runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
        this->bind_child("mlp", mlp_);
        if ("linear_attention" == layer_type_) {
            linear_attn_ = std::make_shared<Qwen3NextGatedDeltaNet>(
                runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("linear_attn", linear_attn_);
        } else {
            self_attn_ = std::make_shared<Qwen3NextAttention>(
                runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("self_attn", self_attn_);
        }
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> Qwen3NextDecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                                                                                                infinilm::runtime::Tensor &hidden_states,
                                                                                                infinilm::runtime::Tensor &residual,
                                                                                                infinilm::global_state::ForwardContext &forward_context) {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (!hidden_states || hidden_states->device() != runtime.device()
        || (residual && residual->device() != runtime.device())
        || input_layernorm_->weight()->device() != runtime.device()
        || post_attention_layernorm_->weight()->device() != runtime.device()) {
        throw std::invalid_argument("Qwen3NextDecoderLayer: forward Runtime/input/norm device mismatch");
    }
    input_layernorm_->forward_inplace(runtime, hidden_states, residual);
    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states, forward_context);
    } else if ("full_attention" == layer_type_) {
        hidden_states = self_attn_->forward(positions, hidden_states, forward_context);
    }

    post_attention_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = mlp_->forward(runtime, hidden_states);
    return std::make_tuple(hidden_states, residual);
}

} // namespace infinilm::models::qwen3_next
