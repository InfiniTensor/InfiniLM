#include "ernie4_5_decoder_layer.hpp"

#include "infinicore/ops.hpp"

#include <algorithm>

namespace infinilm::models::ernie4_5_moe_vl {

namespace {

size_t min_config_index(const nlohmann::json &value, size_t fallback) {
    if (value.is_array()) {
        size_t result = value.empty() ? fallback : value.at(0).get<size_t>();
        for (const auto &item : value) {
            result = std::min(result, item.get<size_t>());
        }
        return result;
    }
    if (value.is_number_unsigned() || value.is_number_integer()) {
        return value.get<size_t>();
    }
    return fallback;
}

size_t max_config_index(const nlohmann::json &value, size_t fallback) {
    if (value.is_array()) {
        size_t result = value.empty() ? fallback : value.at(0).get<size_t>();
        for (const auto &item : value) {
            result = std::max(result, item.get<size_t>());
        }
        return result;
    }
    if (value.is_number_unsigned() || value.is_number_integer()) {
        return value.get<size_t>();
    }
    return fallback;
}

} // namespace

Ernie4_5DecoderLayer::Ernie4_5DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
                                           const infinicore::Device &device)
    : layer_idx_(layer_idx),
      use_moe_(is_moe_layer(model_config, layer_idx)) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);

    if (use_moe_) {
        mlp_ = this->register_module<Ernie4_5SparseMoeBlock>("mlp", model_config, device, layer_idx);
    } else {
        mlp_ = this->register_module<infinilm::layers::MLP>("mlp", model_config, device);
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Ernie4_5DecoderLayer::forward(const infinicore::Tensor &positions,
                                                                                 infinicore::Tensor &hidden_states,
                                                                                 infinicore::Tensor &residual) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = forward_mlp_(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Ernie4_5DecoderLayer::forward(const infinicore::Tensor &positions,
                                                 infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = forward_mlp_(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

infinicore::Tensor Ernie4_5DecoderLayer::forward_mlp_(const infinicore::Tensor &hidden_states) const {
    if (use_moe_) {
        return std::static_pointer_cast<Ernie4_5SparseMoeBlock>(mlp_)->forward(hidden_states);
    }
    return std::static_pointer_cast<infinilm::layers::MLP>(mlp_)->forward(hidden_states);
}

bool Ernie4_5DecoderLayer::is_moe_layer(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                        size_t layer_idx) {
    const auto &config_json = model_config->get_config_json();
    const bool use_moe = model_config->get_or<bool>("use_moe", true);
    if (!use_moe) {
        return false;
    }

    const size_t interval = model_config->get_or<size_t>("moe_layer_interval", 1);
    const size_t start = config_json.contains("moe_layer_start_index")
                           ? min_config_index(config_json.at("moe_layer_start_index"), 0)
                           : 0;
    const size_t end = config_json.contains("moe_layer_end_index")
                         ? max_config_index(config_json.at("moe_layer_end_index"), model_config->get<size_t>("num_hidden_layers") - 1)
                         : model_config->get<size_t>("num_hidden_layers") - 1;

    return interval != 0 && ((layer_idx + 1) % interval == 0) && layer_idx >= start && layer_idx <= end;
}

} // namespace infinilm::models::ernie4_5_moe_vl
