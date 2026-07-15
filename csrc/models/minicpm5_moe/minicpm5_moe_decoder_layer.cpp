#include "minicpm5_moe_decoder_layer.hpp"

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoeDecoderLayer::MiniCPM5MoeDecoderLayer(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    input_layernorm_ =
        this->register_module<infinicore::nn::RMSNorm>("input_layernorm", hidden_size, rms_norm_eps, dtype, device);
    post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>(
        "post_attention_layernorm", hidden_size, rms_norm_eps, dtype, device);
    self_attn_ = this->register_module<MiniCPM5MoeAttention>("self_attn", model_config, layer_idx, device);

    const size_t first_k_dense_replace = model_config->get_or<size_t>("first_k_dense_replace", 0);
    if (layer_idx < first_k_dense_replace) {
        dense_mlp_ = this->register_module<MiniCPM5DenseMLP>("mlp", model_config, device);
    } else {
        moe_mlp_ = this->register_module<MiniCPM5MoeSparseMoeBlock>("mlp", model_config, layer_idx, device);
    }
}

infinicore::Tensor MiniCPM5MoeDecoderLayer::mlp_forward(const infinicore::Tensor &hidden_states) const {
    return dense_mlp_ ? dense_mlp_->forward(hidden_states) : moe_mlp_->forward(hidden_states);
}

infinicore::Tensor MiniCPM5MoeDecoderLayer::mlp_forward_matmul_only(
    const infinicore::Tensor &hidden_states) const {
    if (dense_mlp_) {
        return dense_mlp_->forward_matmul_only(hidden_states);
    }
    return moe_mlp_->forward_matmul_only(hidden_states);
}

void MiniCPM5MoeDecoderLayer::mlp_allreduce_output(infinicore::Tensor &output) const {
    if (dense_mlp_) {
        dense_mlp_->allreduce_output(output);
        return;
    }
    moe_mlp_->allreduce_output(output);
}

std::tuple<infinicore::Tensor, infinicore::Tensor> MiniCPM5MoeDecoderLayer::forward(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual) {
    input_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = self_attn_->forward(positions, hidden_states);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    hidden_states = mlp_forward(hidden_states);
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor MiniCPM5MoeDecoderLayer::forward(const infinicore::Tensor &positions,
                                                    infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    hidden_states = self_attn_->forward(positions, hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

void MiniCPM5MoeDecoderLayer::piecewise_pre_attn(
    const infinicore::Tensor &positions,
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    if (global_state::piecewise_inductor_segment_enabled() &&
        global_state::get_forward_context().piecewise.allow_inductor_pre_attn) {
        const size_t bucket = hidden_states->size(1);
        if (infinicore::op::inductor_segment_impl::has_package(
                infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer_idx_, bucket)) {
            infinicore::op::inductor_segment_(
                positions,
                hidden_states,
                residual,
                staging.q_rope,
                staging.k_rope,
                staging.v_rope,
                infinicore::op::PiecewiseInductorSegmentId::PreAttn,
                layer_idx_,
                bucket);
            if (!infinicore::context::isGraphRecording()) {
                infinicore::context::syncDevice();
            }
            return;
        }
    }
    input_layernorm_->forward_inplace(hidden_states, residual);
    self_attn_->forward_pre_attn_piecewise(positions, hidden_states, staging);
}

void MiniCPM5MoeDecoderLayer::piecewise_post_attn_cg(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    self_attn_->forward_post_attn_piecewise_cg_into(hidden_states, staging);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    auto mlp_out = mlp_forward(hidden_states);
    hidden_states->copy_from(mlp_out);
}

void MiniCPM5MoeDecoderLayer::piecewise_post_attn_graph(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    self_attn_->forward_post_attn_piecewise_graph_into(hidden_states, staging);
    post_attention_layernorm_->forward_inplace(hidden_states, residual);
    auto mlp_out = mlp_forward_matmul_only(hidden_states);
    hidden_states->copy_from(mlp_out);
}

void MiniCPM5MoeDecoderLayer::piecewise_post_attn_allreduce(
    infinicore::Tensor &hidden_states,
    infinicore::Tensor &residual,
    global_state::PiecewiseLayerStaging &staging) const {
    (void)residual;
    self_attn_->forward_post_attn_piecewise_allreduce_into(hidden_states, staging);
    auto &piecewise = global_state::get_forward_context().piecewise;
    const size_t bucket = hidden_states->size(1);
    const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : bucket;
    global_state::ar_profile::allreduce_hidden_valid_contiguous(
        hidden_states,
        valid_len,
        piecewise.ar_staging,
        [&](infinicore::Tensor &t) { mlp_allreduce_output(t); });
}

} // namespace infinilm::models::minicpm5_moe
