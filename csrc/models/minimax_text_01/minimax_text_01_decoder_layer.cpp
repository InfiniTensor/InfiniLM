#include "minimax_text_01_decoder_layer.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/mul_scalar.hpp"

#include <stdexcept>
#include <vector>

namespace infinilm::models::minimax_text_01 {

MiniMaxText01DecoderLayer::MiniMaxText01DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                     size_t layer_idx,
                                                     const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    // Every layer has both layer norms and the MoE feed-forward block.
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);

    // `attn_type_list` decides the attention module. From the reference
    // `modeling_minimax_text_01.py`: 0 = linear attention (Lightning),
    // 1 = full attention. Only full attention is supported for now.
    const std::vector<int> attn_type_list = model_config->get<std::vector<int>>("attn_type_list");
    attn_type_ = attn_type_list[layer_idx];
    if (1 == attn_type_) {
        INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    } else if (0 == attn_type_) {
        INFINICORE_NN_MODULE_INIT(linear_attn, model_config, layer_idx, device);
    } else {
        throw std::runtime_error(
            "infinilm::models::minimax_text_01::MiniMaxText01DecoderLayer: unsupported attn_type '"
            + std::to_string(attn_type_) + "' for layer " + std::to_string(layer_idx));
    }

    // MiniMax combines the residual path with the sublayer output using
    // learnable per-block alpha/beta scaling. The attention block picks the
    // full/linear alpha/beta by layer type; the MoE block always uses the MLP
    // alpha/beta.
    postnorm_ = model_config->get_or<bool>("postnorm", false);
    if (0 == attn_type_) {
        layernorm_attention_alpha_ = model_config->get_or<double>("layernorm_linear_attention_alpha", 1.0);
        layernorm_attention_beta_ = model_config->get_or<double>("layernorm_linear_attention_beta", 1.0);
    } else {
        layernorm_attention_alpha_ = model_config->get_or<double>("layernorm_full_attention_alpha", 1.0);
        layernorm_attention_beta_ = model_config->get_or<double>("layernorm_full_attention_beta", 1.0);
    }
    layernorm_mlp_alpha_ = model_config->get_or<double>("layernorm_mlp_alpha", 1.0);
    layernorm_mlp_beta_ = model_config->get_or<double>("layernorm_mlp_beta", 1.0);
}

// Plain forward matching `MiniMaxText01DecoderLayer.forward` in the reference
// `modeling_minimax_text_01.py`. With `postnorm`, the residual is the normed
// input; each block output is `residual * alpha + sublayer_output * beta`.
infinicore::Tensor MiniMaxText01DecoderLayer::forward(const infinicore::Tensor &positions,
                                                      infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    if (postnorm_) {
        residual = hidden_states;
    }
    if (1 == attn_type_) {
        hidden_states = self_attn_->forward(positions, hidden_states);
    } else {
        hidden_states = linear_attn_->forward(hidden_states);
    }
    hidden_states = infinicore::op::add(
        infinicore::op::mul_scalar(residual, layernorm_attention_alpha_),
        infinicore::op::mul_scalar(hidden_states, layernorm_attention_beta_));

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    if (postnorm_) {
        residual = hidden_states;
    }
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = infinicore::op::add(
        infinicore::op::mul_scalar(residual, layernorm_mlp_alpha_),
        infinicore::op::mul_scalar(hidden_states, layernorm_mlp_beta_));
    return hidden_states;
}

} // namespace infinilm::models::minimax_text_01
