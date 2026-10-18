#include "ernie4_5_decoder_layer.hpp"
#include "../models_registry.hpp"

#include "../../global_state/forward_context.hpp"

#include <algorithm>
#include <utility>

#include <exception>

namespace infinilm::models::ernie4_5_vl {
namespace {

size_t min_json_size(const nlohmann::json &value, size_t fallback) {
    if (value.is_array() && !value.empty()) {
        size_t result = value.at(0).get<size_t>();
        for (const auto &item : value) {
            result = std::min(result, item.get<size_t>());
        }
        return result;
    }
    return value.is_number_unsigned() ? value.get<size_t>() : fallback;
}

size_t max_json_size(const nlohmann::json &value, size_t fallback) {
    if (value.is_array() && !value.empty()) {
        size_t result = value.at(0).get<size_t>();
        for (const auto &item : value) {
            result = std::max(result, item.get<size_t>());
        }
        return result;
    }
    return value.is_number_unsigned() ? value.get<size_t>() : fallback;
}

bool is_moe_layer(const nlohmann::json &config, size_t layer_idx) {
    const size_t interval = config.value("moe_layer_interval", 1);
    const size_t start = config.contains("moe_layer_start_index") ? min_json_size(config.at("moe_layer_start_index"), 0) : 0;
    const size_t end = config.contains("moe_layer_end_index") ? max_json_size(config.at("moe_layer_end_index"), config.value("num_hidden_layers", 1) - 1) : config.value("num_hidden_layers", 1) - 1;
    return config.value("use_moe", false) && interval > 0 && ((layer_idx + 1) % interval == 0) && layer_idx >= start && layer_idx <= end;
}

} // namespace

Ernie45DecoderLayer::Ernie45DecoderLayer(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         std::shared_ptr<const Ernie45MropeCache> mrope_cache,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        input_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*input_layernorm_)>>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("input_layernorm", input_layernorm_);
        post_attention_layernorm_ = std::make_shared<std::remove_reference_t<decltype(*post_attention_layernorm_)>>(runtime, hidden_size, rms_norm_eps, dtype);
        this->bind_child("post_attention_layernorm", post_attention_layernorm_);
        self_attn_ = std::make_shared<std::remove_reference_t<decltype(*self_attn_)>>(runtime, model_config, layer_idx, std::move(mrope_cache), rank_info, attention_backend);
        this->bind_child("self_attn", self_attn_);

    use_moe_ = is_moe_layer(model_config->get_config_json(), layer_idx);
    if (use_moe_) {
            mlp_ = std::make_shared<Ernie45MoE>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("mlp", mlp_);
            // ERNIE also defines an unregistered mlp_text path for pure text tokens when multimodal
            // experts are enabled. This first implementation routes all tokens through mlp.
    } else {
            mlp_ = std::make_shared<infinilm::layers::mlp::MLP>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
            this->bind_child("mlp", mlp_);
        }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> Ernie45DecoderLayer::forward(const infinilm::runtime::Tensor &positions,
                                                                                infinilm::runtime::Tensor &hidden_states,
                                                                                infinilm::runtime::Tensor &residual,
                                                                                global_state::ForwardContext &forward_context) {
    auto &runtime = forward_context.require_runtime();
    input_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states, forward_context);

    post_attention_layernorm_->forward_inplace(runtime, hidden_states, residual);
    hidden_states = use_moe_
                      ? std::static_pointer_cast<Ernie45MoE>(mlp_)->forward(hidden_states, forward_context)
                      : std::static_pointer_cast<infinilm::layers::mlp::MLP>(mlp_)->forward(runtime, hidden_states);
    return {hidden_states, residual};
}

} // namespace infinilm::models::ernie4_5_vl
