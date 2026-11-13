#include "minicpm5_moe_decoder_layer.hpp"

#include "../../global_state/ar_profile.hpp"
#include "../../global_state/decode_phase_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../global_state/piecewise_inductor_flags.hpp"
#include "infinicore/context/context.hpp"

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoEDecoderLayer::MiniCPM5MoEDecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    layer_idx_ = layer_idx;
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

infinicore::Tensor MiniCPM5MoEDecoderLayer::mlp_forward(
    const infinicore::Tensor &hidden_states) const {
    return use_moe_ ? moe_mlp_->forward(hidden_states) : dense_mlp_->forward(hidden_states);
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
MiniCPM5MoEDecoderLayer::forward(const infinicore::Tensor &positions,
                                 infinicore::Tensor &hidden_states,
                                 infinicore::Tensor &residual) const {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = mlp_forward(hidden_states);
    return {hidden_states, residual};
}

void MiniCPM5MoEDecoderLayer::piecewise_pre_attn(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    (void)global_state::piecewise_inductor_segment_enabled();
    piecewise_pre_attn_qkv(positions, hidden_states, residual, staging);
    piecewise_pre_attn_rope(positions, staging);
}

void MiniCPM5MoEDecoderLayer::piecewise_pre_attn_qkv(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    input_layernorm_->forward_inplace(hidden_states, residual);
    self_attn_->forward_pre_attn_qkv_piecewise(positions, hidden_states, staging);
}

void MiniCPM5MoEDecoderLayer::piecewise_pre_attn_rope(
    const infinicore::Tensor &positions,
    global_state::PiecewiseLayerStaging &staging) const {
    self_attn_->forward_pre_attn_rope_piecewise(positions, staging);
}

void MiniCPM5MoEDecoderLayer::piecewise_post_attn_cg(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    self_attn_->forward_post_attn_piecewise_cg_into(hidden_states, staging);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    auto mlp_out = mlp_forward(hidden_states);
    hidden_states->copy_from(mlp_out);
}

void MiniCPM5MoEDecoderLayer::piecewise_post_attn_decode_cg(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    self_attn_->forward_post_attn_piecewise_cg_into(hidden_states, staging);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    if (!use_moe_) {
        auto mlp_out = dense_mlp_->forward(hidden_states);
        hidden_states->copy_from(mlp_out);
    } else if (infinicore::context::moeTritonCaptureAllowed()) {
        auto mlp_out = moe_mlp_->forward(hidden_states);
        hidden_states->copy_from(mlp_out);
    }
}

void MiniCPM5MoEDecoderLayer::piecewise_eager_moe(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    (void)residual;
    (void)staging;
    if (!use_moe_) {
        return;
    }
    if (infinicore::context::moeTritonCaptureAllowed()) {
        return;
    }
    auto mlp_out = moe_mlp_->forward(hidden_states);
    hidden_states->copy_from(mlp_out);
}

void MiniCPM5MoEDecoderLayer::piecewise_post_attn_graph(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    // Without deferred allreduce split, reuse the fused CG post path.
    piecewise_post_attn_cg(hidden_states, residual, staging);
}

void MiniCPM5MoEDecoderLayer::piecewise_post_attn_allreduce(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    (void)residual;
    self_attn_->forward_post_attn_piecewise_allreduce_into(hidden_states, staging);
}

} // namespace infinilm::models::minicpm5_moe
