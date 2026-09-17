#include "granitemoehybrid_decoderLayer.hpp"

#include "infinicore/ops.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::granitemoehybrid {

GraniteMoeHybridDecoderLayer::GraniteMoeHybridDecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx),
      has_experts_(model_config->get_or<size_t>("num_local_experts", 0) > 0) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(shared_mlp, model_config, device);
    if (has_experts_) {
        INFINICORE_NN_MODULE_INIT(block_sparse_moe, model_config, device);
    }

    const auto layer_types = model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types.at(layer_idx);
    if ("mamba" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(mamba, model_config, layer_idx, device);
    } else if ("attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    } else {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridDecoderLayer: "
            "unsupported layer_type '" +
            layer_type_ + "' for layer " + std::to_string(layer_idx));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> GraniteMoeHybridDecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual) const {
    input_layernorm_->forward_inplace(hidden_states, residual);
    if ("mamba" == layer_type_) {
        hidden_states = mamba_->forward(hidden_states);
    } else {
        hidden_states = self_attn_->forward(positions, hidden_states);
    }

    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    if (has_experts_) {
        auto moe_hidden_states = block_sparse_moe_->forward(hidden_states);
        auto shared_hidden_states = shared_mlp_->forward(hidden_states);
        hidden_states = infinicore::op::add(moe_hidden_states, shared_hidden_states);
    } else {
        hidden_states = shared_mlp_->forward(hidden_states);
    }
    return {hidden_states, residual};
}

infinicore::Tensor GraniteMoeHybridDecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states) const {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    if ("mamba" == layer_type_) {
        hidden_states = mamba_->forward(hidden_states);
    } else {
        hidden_states = self_attn_->forward(positions, hidden_states);
    }
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    if (has_experts_) {
        auto moe_hidden_states = block_sparse_moe_->forward(hidden_states);
        auto shared_hidden_states = shared_mlp_->forward(hidden_states);
        hidden_states = infinicore::op::add(moe_hidden_states, shared_hidden_states);
    } else {
        hidden_states = shared_mlp_->forward(hidden_states);
    }
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

} // namespace infinilm::models::granitemoehybrid
