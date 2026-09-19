#include "minimax_decoderLayer.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/mul_scalar.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::minimax {

MiniMaxDecoderLayer::MiniMaxDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get_or<double>("rms_norm_eps", 1e-5);

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);

    const std::vector<std::string> layer_types = model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types.at(layer_idx);
    if ("linear_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(linear_attn, model_config, layer_idx, device);
    } else if ("full_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    } else {
        throw std::runtime_error("infinilm::models::minimax::MiniMaxDecoderLayer: unsupported layer_type '" + layer_type_ + "' for layer " + std::to_string(layer_idx));
    }

    num_experts_ = model_config->get_or<size_t>("num_experts", 1);
    if (num_experts_ > 1) {
        INFINICORE_NN_MODULE_INIT(moe, model_config, layer_idx, device);
    } else {
        INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
    }

    alpha_attn_ = model_config->get_or("layernorm_attention_alpha", model_config->get_or("linear_attn_alpha_factor", 1.0));
    beta_attn_ = model_config->get_or("layernorm_attention_beta", model_config->get_or("linear_attn_beta_factor", 1.0));
    alpha_mlp_ = model_config->get_or("layernorm_mlp_alpha", model_config->get_or("mlp_alpha_factor", 1.0));
    beta_mlp_ = model_config->get_or("layernorm_mlp_beta", model_config->get_or("mlp_beta_factor", 1.0));
}

infinicore::Tensor MiniMaxDecoderLayer::forward(const infinicore::Tensor &positions,
                                                infinicore::Tensor &hidden_states) const {
    // HF transformers `minimax` carries the *normalized* value in both residual
    // paths: x1 = norm(x0); out_attn = x1 + attn(x1); x2 = norm(out_attn);
    // out = x2 + mlp(x2).
    auto x = input_layernorm_->forward(hidden_states);
    auto residual = x;
    if ("linear_attention" == layer_type_) {
        x = linear_attn_->forward(x);
    } else {
        x = self_attn_->forward(positions, x);
    }
    auto scale_add = [](const infinicore::Tensor &a, double sa,
                        const infinicore::Tensor &b, double sb) -> infinicore::Tensor {
        if (sa == 1.0 && sb == 1.0) {
            return infinicore::op::add(a, b);
        }
        auto a_scaled = sa == 1.0 ? a : infinicore::op::mul_scalar(a, sa);
        auto b_scaled = sb == 1.0 ? b : infinicore::op::mul_scalar(b, sb);
        return infinicore::op::add(a_scaled, b_scaled);
    };
    auto post_input = scale_add(residual, alpha_attn_, x, beta_attn_);

    // Pre-norm MLP sub-block (residual = normalized post-attention value).
    auto normed = post_attention_layernorm_->forward(post_input);
    auto mlp_out = num_experts_ > 1 ? moe_->forward(normed) : mlp_->forward(normed);
    return scale_add(normed, alpha_mlp_, mlp_out, beta_mlp_);
}

} // namespace infinilm::models::minimax


