#include "minicpm5_moe_decoder_layer.hpp"

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoEDecoderLayer::MiniCPM5MoEDecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);

    const size_t first_dense_layers = model_config->get_or<size_t>("first_k_dense_replace", 0);
    use_moe_ = model_config->get_or<size_t>("n_routed_experts", 0) > 0
            && layer_idx >= first_dense_layers;
    if (use_moe_) {
        moe_mlp_ = this->register_module<MiniCPM5MoESparseMoeBlock>(
            "mlp", model_config, layer_idx, device);
    } else {
        auto dense_config = make_minicpm5_mlp_config(
            model_config, model_config->get<size_t>("intermediate_size"));
        dense_mlp_ = this->register_module<infinilm::layers::mlp::MLP>(
            "mlp", dense_config, device);
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
MiniCPM5MoEDecoderLayer::forward(const infinicore::Tensor &positions,
                                 infinicore::Tensor &hidden_states,
                                 infinicore::Tensor &residual) const {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = use_moe_ ? moe_mlp_->forward(hidden_states)
                             : dense_mlp_->forward(hidden_states);
    return {hidden_states, residual};
}

} // namespace infinilm::models::minicpm5_moe
