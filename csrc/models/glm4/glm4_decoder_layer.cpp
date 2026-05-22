#include "glm4_decoder_layer.hpp"

namespace infinilm::models::glm4 {

Glm4DecoderLayer::Glm4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t layer_idx,
                                   const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_self_attn_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_mlp_layernorm, hidden_size, rms_norm_eps, dtype, device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Glm4DecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual) {

    // 1. Attention Block
    residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = post_self_attn_layernorm_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    // 2. MLP Block
    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = post_mlp_layernorm_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Glm4DecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states) {

    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    // hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = post_self_attn_layernorm_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = post_mlp_layernorm_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    return hidden_states;
}

} // namespace infinilm::models::glm4
