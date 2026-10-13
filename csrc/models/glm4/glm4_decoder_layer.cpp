#include "execution/submit.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "glm4_decoder_layer.hpp"
#include "runtime/runtime.hpp"
#include "../models_registry.hpp"
#include <exception>

namespace infinilm::models::glm4 {

Glm4DecoderLayer::Glm4DecoderLayer(infinilm::runtime::Runtime &runtime,
                                   std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t layer_idx,
                                   const engine::distributed::RankInfo &rank_info,
                                   backends::AttentionBackend attention_backend,
                                   const layers::moe::EPConfig &ep_config) {
    infinilm::models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    const auto &dtype = model_config->get_dtype();
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    try {
        self_attn_ = std::make_shared<Glm4Attention>(
            runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
        this->bind_child("self_attn", self_attn_);
        mlp_ = std::make_shared<infinilm::layers::mlp::MLP>(
            runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
        this->bind_child("mlp", mlp_);
        input_layernorm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("input_layernorm", input_layernorm_);
        post_attention_layernorm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("post_attention_layernorm", post_attention_layernorm_);
        post_self_attn_layernorm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("post_self_attn_layernorm", post_self_attn_layernorm_);
        post_mlp_layernorm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("post_mlp_layernorm", post_mlp_layernorm_);
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) runtime.syncStream();
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
        }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> Glm4DecoderLayer::forward(
    const infinilm::runtime::Tensor &positions,
    infinilm::runtime::Tensor &hidden_states,
    infinilm::runtime::Tensor &residual,
    infinilm::global_state::ForwardContext &forward_context) {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();

    // 1. Attention Block
    residual = hidden_states;
    hidden_states = input_layernorm_->forward(runtime, hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states, forward_context);
    hidden_states = post_self_attn_layernorm_->forward(runtime, hidden_states);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, residual->shape(), residual->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), residual, hidden_states, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        hidden_states = std::move(ops_output);
    }

    // 2. MLP Block
    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(runtime, hidden_states);
    hidden_states = mlp_->forward(runtime, hidden_states);
    hidden_states = post_mlp_layernorm_->forward(runtime, hidden_states);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, residual->shape(), residual->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), residual, hidden_states, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        hidden_states = std::move(ops_output);
    }

    return std::make_tuple(hidden_states, residual);
}


} // namespace infinilm::models::glm4
