#include "ernie4_5_decoder_layer.hpp"

#include "infinicore/ops.hpp"

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5DecoderLayer::Ernie4_5DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
                                           const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);

    if (layer_idx == 0) {
        dense_mlp_ = this->register_module<infinilm::layers::mlp::MLP>("mlp", model_config, device);
    } else {
        moe_mlp_ = this->register_module<Ernie4_5TextMoeBlock>("mlp", model_config, device);
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
Ernie4_5DecoderLayer::forward(const infinicore::Tensor &positions,
                              infinicore::Tensor &hidden_states,
                              infinicore::Tensor &residual,
                              const infinicore::Tensor &token_type_ids) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = dense_mlp_ ? dense_mlp_->forward(hidden_states) : moe_mlp_->forward(hidden_states, token_type_ids);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Ernie4_5DecoderLayer::forward(const infinicore::Tensor &positions,
                                                 infinicore::Tensor &hidden_states,
                                                 const infinicore::Tensor &token_type_ids) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = dense_mlp_ ? dense_mlp_->forward(hidden_states) : moe_mlp_->forward(hidden_states, token_type_ids);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

} // namespace infinilm::models::ernie4_5_moe_vl
