#include "minicpm_sala_decoderLayer.hpp"

#include "infinicore/ops.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::minicpm_sala {

MiniCPMSALADecoderLayer::MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 size_t layer_idx,
                                                 const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t intermediate_size = model_config->get<size_t>("intermediate_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);

    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);

    std::vector<std::string> mixer_types = model_config->get<std::vector<std::string>>("mixer_types");
    std::string mixer_type = mixer_types[layer_idx];
    if ("minicpm4" == mixer_type) {
        self_attn_ = std::make_shared<MiniCPMSALAAttention>(this->register_module<InfLLMv2Attention>("self_attn", model_config, layer_idx, device));
    } else if ("lightning" == mixer_type || "lightning_attn" == mixer_type || "lightning-attn" == mixer_type) {
        self_attn_ = std::make_shared<MiniCPMSALAAttention>(this->register_module<LightningAttention>("self_attn", model_config, layer_idx, device));
    } else {
        throw std::runtime_error("infinilm::models::minicpm_sala::MiniCPMSALADecoderLayer: unsupported mixer_type '" + mixer_type + "' for layer " + std::to_string(layer_idx));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> MiniCPMSALADecoderLayer::forward(infinicore::Tensor &hidden_states, infinicore::Tensor &residual) {

    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = std::visit(
        [&](auto &attn_ptr) { return attn_ptr->forward(hidden_states); }, *self_attn_);

    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = mlp_->forward(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor MiniCPMSALADecoderLayer::forward(infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;

    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = std::visit(
        [&](auto &attn_ptr) { return attn_ptr->forward(hidden_states); }, *self_attn_);

    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

void MiniCPMSALADecoderLayer::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    if (self_attn_) {
        std::visit([&](auto &attn_ptr) { attn_ptr->set_rotary_emb(rotary_emb); }, *self_attn_);
    }
}

} // namespace infinilm::models::minicpm_sala
