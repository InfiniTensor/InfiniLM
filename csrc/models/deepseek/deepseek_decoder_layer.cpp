#include "deepseek_decoder_layer.hpp"

#include "infinicore/ops.hpp"

namespace infinilm::models::deepseek {

DeepseekDecoderLayer::DeepseekDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           size_t layer_idx,
                                           const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);

    const size_t first_k_dense_replace = model_config->get_or<size_t>("first_k_dense_replace", 0);
    const size_t moe_layer_freq = model_config->get_or<size_t>("moe_layer_freq", 1);
    const bool use_moe = model_config->get_or<size_t>("n_routed_experts", 0) > 0
                      && layer_idx >= first_k_dense_replace
                      && (moe_layer_freq == 0 || layer_idx % moe_layer_freq == 0);

    if (use_moe) {
        mlp_ = std::make_shared<DeepseekMLP>(
            this->register_module<deepseek_v2::DeepseekV2MoE>("mlp", model_config, device));
    } else {
        mlp_ = std::make_shared<DeepseekMLP>(
            this->register_module<deepseek_v2::DeepseekV2MLP>("mlp", model_config, device));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
DeepseekDecoderLayer::forward(const infinicore::Tensor &positions,
                              infinicore::Tensor &hidden_states,
                              infinicore::Tensor &residual) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = std::visit(
        [&](auto &mlp_ptr) { return mlp_ptr->forward(hidden_states); }, *mlp_);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor DeepseekDecoderLayer::forward(const infinicore::Tensor &positions,
                                                 infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = std::visit(
        [&](auto &mlp_ptr) { return mlp_ptr->forward(hidden_states); }, *mlp_);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

} // namespace infinilm::models::deepseek
