#include "qwen3_next_decoderLayer.hpp"

#include "infinicore/ops.hpp"
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextDecoderLayer::Qwen3NextDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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

    const std::vector<std::string> layer_types = model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types[layer_idx];
    if ("linear_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(linear_attn, model_config, layer_idx, device);
    } else if ("full_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    } else {
        throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextDecoderLayer: unsupported layer_type '" + layer_type_ + "' for layer " + std::to_string(layer_idx));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Qwen3NextDecoderLayer::forward(infinicore::Tensor &hidden_states, infinicore::Tensor &residual) {

    input_layernorm_->forward_inplace(hidden_states, residual);

    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states);
    } else if ("full_attention" == layer_type_) {
        hidden_states = self_attn_->forward(hidden_states);
    }

    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = mlp_->forward(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Qwen3NextDecoderLayer::forward(infinicore::Tensor &hidden_states) {

    auto residual = hidden_states;

    hidden_states = input_layernorm_->forward(hidden_states);
    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states);
    } else if ("full_attention" == layer_type_) {
        hidden_states = self_attn_->forward(hidden_states);
    }
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

void Qwen3NextDecoderLayer::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    if (self_attn_) {
        self_attn_->set_rotary_emb(rotary_emb);
    }
}

} // namespace infinilm::models::qwen3_next
