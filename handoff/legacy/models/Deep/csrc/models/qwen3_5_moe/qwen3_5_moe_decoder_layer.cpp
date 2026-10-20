#include "qwen3_5_moe_decoder_layer.hpp"

#include "runtime/runtime.hpp"

#include "runtime/runtime.hpp"
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5_moe {

Qwen35MoeDecoderLayer::Qwen35MoeDecoderLayer(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config)
    : layer_idx_(layer_idx) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Qwen35 decoder construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Qwen35 decoder construction resources");
    const auto layer_types = model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types.at(layer_idx);
    if (layer_type_ != "linear_attention" && layer_type_ != "full_attention")
        throw std::invalid_argument("Qwen35 decoder has unsupported layer_type");
    try {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        input_layernorm_ = this->bind_child("input_layernorm", std::make_shared<std::remove_reference_t<decltype(*input_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        post_attention_layernorm_ = this->bind_child("post_attention_layernorm", std::make_shared<std::remove_reference_t<decltype(*post_attention_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        mlp_ = this->bind_child("mlp", std::make_shared<std::remove_reference_t<decltype(*mlp_ )>>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));

    if ("linear_attention" == layer_type_) {
            linear_attn_ = this->bind_child("linear_attn", std::make_shared<std::remove_reference_t<decltype(*linear_attn_ )>>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));
    } else if ("full_attention" == layer_type_) {
            self_attn_ = this->bind_child("self_attn", std::make_shared<std::remove_reference_t<decltype(*self_attn_ )>>(runtime, model_config, layer_idx, rank_info, attention_backend));
    } else {
        throw std::runtime_error(
            "Qwen35MoeDecoderLayer: unsupported layer_type '" + layer_type_
            + "' for layer " + std::to_string(layer_idx));
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> Qwen35MoeDecoderLayer::forward(
    const infinilm::runtime::Tensor &positions,
    infinilm::runtime::Tensor &hidden_states,
    infinilm::runtime::Tensor &residual,
    infinilm::global_state::ForwardContext &forward_context) {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (!hidden_states || hidden_states->device() != runtime.device()
        || (residual && residual->device() != runtime.device())
        || input_layernorm_->weight()->device() != runtime.device()
        || post_attention_layernorm_->weight()->device() != runtime.device()) {
        throw std::invalid_argument("Qwen35MoeDecoderLayer: forward Runtime/input/norm device mismatch");
    }
    input_layernorm_->forward_inplace(runtime, hidden_states, residual);
    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states, forward_context);
    } else {
        hidden_states = self_attn_->forward(positions, hidden_states, forward_context);
    }

    post_attention_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = mlp_->forward(runtime, hidden_states);
    return std::make_tuple(hidden_states, residual);
}


} // namespace infinilm::models::qwen3_5_moe
