#include "glm4_decoder_layer.hpp"
#include "infinicore/ops.hpp" // 包含 add 等算子

namespace infinilm::models::glm4 {

Glm4DecoderLayer::Glm4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   size_t layer_idx,
                                   const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    self_attn_ = this->register_module<Glm4Attention>(
        "self_attn", model_config, layer_idx, device);

    mlp_ = this->register_module<infinilm::layers::mlp::MLP>(
        "mlp", model_config, device); 

    input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>(
        "input_layernorm", hidden_size, rms_norm_eps, dtype, device);
    
    post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>(
        "post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);

    post_self_attn_layernorm_ = this->register_module<infinicore::nn::RMSNorm>(
        "post_self_attn_layernorm", hidden_size, rms_norm_eps, dtype, device);

    post_mlp_layernorm_ = this->register_module<infinicore::nn::RMSNorm>(
        "post_mlp_layernorm", hidden_size, rms_norm_eps, dtype, device);
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
    //hidden_states = post_attention_layernorm_->forward(hidden_states);
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

