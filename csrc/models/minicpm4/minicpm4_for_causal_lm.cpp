#include "execution/submit.hpp"
#include "minicpm4_for_causal_lm.hpp"
#include "../../global_state/global_state.hpp"
#include "runtime/runtime.hpp"
#include "../models_registry.hpp"

#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm4 {

namespace {
float residual_scale(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    const float scale_depth = model_config->get_or<float>("scale_depth", 1.0f);
    if (model_config->get_or<std::string>("model_type", "") == "minicpm_eagle") {
        const float mup_denominator = model_config->get_or<float>("mup_denominator", 1.0f);
        return scale_depth / std::sqrt(mup_denominator);
    }
    const float num_hidden_layers = static_cast<float>(model_config->get<size_t>("num_hidden_layers"));
    return scale_depth / std::sqrt(num_hidden_layers);
}
} // namespace

MiniCPM4Attention::MiniCPM4Attention(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config)
    : Attention(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config) {
    try {
    o_proj_->set_alpha(residual_scale(model_config));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

MiniCPM4MLP::MiniCPM4MLP(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config)
    : MLP(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config) {
    try {
    down_proj_->set_alpha(residual_scale(model_config));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

MiniCPM4DecoderLayer::MiniCPM4DecoderLayer(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        input_layernorm_ = this->bind_child("input_layernorm", std::make_shared<std::remove_reference_t<decltype(*input_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        self_attn_ = this->bind_child("self_attn", std::make_shared<std::remove_reference_t<decltype(*self_attn_ )>>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));
        post_attention_layernorm_ = this->bind_child("post_attention_layernorm", std::make_shared<std::remove_reference_t<decltype(*post_attention_layernorm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
        mlp_ = this->bind_child("mlp", std::make_shared<std::remove_reference_t<decltype(*mlp_ )>>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> MiniCPM4DecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                                                                                 infinilm::runtime::Tensor &hidden_states,
                                                                                 infinilm::runtime::Tensor &residual,
                                                                                 global_state::ForwardContext &forward_context) {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    input_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states, forward_context);
    post_attention_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = mlp_->forward(runtime, hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinilm::runtime::Tensor MiniCPM4DecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                                                 infinilm::runtime::Tensor &hidden_states,
                                                 global_state::ForwardContext &forward_context) {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(runtime, hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states, forward_context);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, residual->shape(), residual->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), residual, hidden_states, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        hidden_states = std::move(ops_output);
    }

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(runtime, hidden_states);
    hidden_states = mlp_->forward(runtime, hidden_states);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, residual->shape(), residual->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), residual, hidden_states, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        hidden_states = std::move(ops_output);
    }
    return hidden_states;
}

MiniCPM4Model::MiniCPM4Model(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
    const auto &dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        embed_tokens_ = this->bind_child("embed_tokens", std::make_shared<std::remove_reference_t<decltype(*embed_tokens_ )>>(runtime, vocab_size, hidden_size, std::nullopt, dtype));
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(this->bind_child("layers." + std::to_string(i), std::make_shared<MiniCPM4DecoderLayer>(runtime, model_config, i, rank_info, attention_backend, ep_config)));
    }
        norm_ = this->bind_child("norm", std::make_shared<std::remove_reference_t<decltype(*norm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor MiniCPM4Model::forward(const infinilm::InfinilmModel::Input &input,
                                      global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(runtime, input_ids);

    infinilm::runtime::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(positions, hidden_states, residual, forward_context);
    }

    norm_->forward_inplace(runtime, hidden_states, residual);
    return hidden_states;
}

MiniCPM4ForCausalLM::MiniCPM4ForCausalLM(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

        model_ = this->bind_child("model", std::make_shared<std::remove_reference_t<decltype(*model_ )>>(runtime, model_config, rank_info, attention_backend, ep_config));
        lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_ )>>(runtime, hidden_size, vocab_size, false, dtype));

    if (model_config->get_config_json().contains("dim_model_base")) {
        const float dim_model_base = model_config->get<float>("dim_model_base");
        lm_head_->set_alpha(dim_model_base / static_cast<float>(hidden_size));
    }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::InfinilmModel::Output MiniCPM4ForCausalLM::forward(
    const infinilm::InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto hidden_states = model_->forward(input, forward_context);
    auto logits = lm_head_->forward(runtime, hidden_states);
    return {logits, hidden_states};
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpm4" != model_type && "minicpm" != model_type) {
        throw std::runtime_error("infinilm::models::minicpm4::create_minicpm4_model_config: model_type is not minicpm4");
    }

    auto &json = model_config->get_config_json();
    if (!json.contains("head_dim")) {
        json["head_dim"] = model_config->get<size_t>("hidden_size") / model_config->get<size_t>("num_attention_heads");
    }
    if (!json.contains("rope_theta")) {
        json["rope_theta"] = 10000.0;
    }
    if (json.contains("bias")) {
        json["attention_bias"] = json["bias"];
        json["mlp_bias"] = json["bias"];
    }
    if (!json.contains("attention_bias")) {
        json["attention_bias"] = false;
    }
    if (!json.contains("mlp_bias")) {
        json["mlp_bias"] = false;
    }
    if (!json.contains("attention_output_bias")) {
        json["attention_output_bias"] = false;
    }
    return model_config;
}

} // namespace infinilm::models::minicpm4

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    minicpm4,
    infinilm::models::minicpm4::MiniCPM4ForCausalLM,
    infinilm::models::minicpm4::create_minicpm4_model_config);
} // namespace
