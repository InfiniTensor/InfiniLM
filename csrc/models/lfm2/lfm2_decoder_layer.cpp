#include "lfm2_decoder_layer.hpp"

#include <infinicore/ops/add.hpp>

#include <stdexcept>
#include <vector>

namespace infinilm::models::lfm2 {

Lfm2DecoderLayer::Lfm2DecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(
        operator_norm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(
        ffn_norm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(feed_forward, model_config, device);

    const auto layer_types =
        model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types.at(layer_idx_);
    if (layer_type_ == "full_attention") {
        INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx_, device);
    } else if (layer_type_ == "short_conv") {
        INFINICORE_NN_MODULE_INIT(conv, model_config, layer_idx_, device);
    } else {
        throw std::runtime_error(
            "Lfm2DecoderLayer: unsupported layer type '" + layer_type_ + "'");
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Lfm2DecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual) {
    operator_norm_->forward_inplace(hidden_states, residual);
    if (layer_type_ == "full_attention") {
        hidden_states = self_attn_->forward(positions, hidden_states);
    } else {
        hidden_states = conv_->forward(hidden_states);
    }

    ffn_norm_->forward_inplace(hidden_states, residual);
    hidden_states = feed_forward_->forward(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Lfm2DecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = operator_norm_->forward(hidden_states);
    if (layer_type_ == "full_attention") {
        hidden_states = self_attn_->forward(positions, hidden_states);
    } else {
        hidden_states = conv_->forward(hidden_states);
    }
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = ffn_norm_->forward(hidden_states);
    hidden_states = feed_forward_->forward(hidden_states);
    return infinicore::op::add(residual, hidden_states);
}

} // namespace infinilm::models::lfm2
