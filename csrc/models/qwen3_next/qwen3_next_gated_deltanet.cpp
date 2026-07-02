#include "qwen3_next_gated_deltanet.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/causal_conv1d.hpp>
#include <infinicore/ops/chunk_gated_delta_rule.hpp>
#include <infinicore/ops/fused_gated_delta_net_gating.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/recurrent_gated_delta_rule.hpp>
#include <infinicore/ops/silu.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextGatedDeltaNet::Qwen3NextGatedDeltaNet(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t linear_num_value_heads = model_config->get<size_t>("linear_num_value_heads");
    size_t linear_num_key_heads = model_config->get<size_t>("linear_num_key_heads");
    size_t linear_key_head_dim = model_config->get<size_t>("linear_key_head_dim");
    size_t linear_value_head_dim = model_config->get<size_t>("linear_value_head_dim");

    linear_num_value_heads_ = linear_num_value_heads;
    linear_num_key_heads_ = linear_num_key_heads;
    linear_key_head_dim_ = linear_key_head_dim;
    linear_value_head_dim_ = linear_value_head_dim;
    key_dim_ = linear_key_head_dim_ * linear_num_key_heads_;
    value_dim_ = linear_value_head_dim_ * linear_num_value_heads_;

    size_t linear_conv_kernel_dim = model_config->get<size_t>("linear_conv_kernel_dim");

    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    size_t conv_dim = key_dim_ * 2 + value_dim_;
    conv_dim_ = conv_dim;
    conv_state_len_ = linear_conv_kernel_dim > 0 ? linear_conv_kernel_dim - 1 : 0;
    conv1d_weight_ = infinicore::nn::Parameter({conv_dim, 1, linear_conv_kernel_dim}, dtype, device);
    this->register_parameter("conv1d.weight", conv1d_weight_);

    size_t projection_size_qkv = key_dim_ * 2 + value_dim_;

    in_proj_qkv_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("in_proj_qkv", hidden_size, projection_size_qkv, false, dtype, device);
    in_proj_z_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("in_proj_z", hidden_size, value_dim_, false, dtype, device);
    in_proj_a_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("in_proj_a", hidden_size, linear_num_value_heads, false, dtype, device);
    in_proj_b_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("in_proj_b", hidden_size, linear_num_value_heads, false, dtype, device);

    INFINICORE_NN_PARAMETER_INIT(dt_bias, ({linear_num_value_heads}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(A_log, ({linear_num_value_heads}, dtype, device));

    INFINICORE_NN_MODULE_INIT(norm, linear_value_head_dim, rms_norm_eps, dtype, device);
    out_proj_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("out_proj", value_dim_, hidden_size, false, dtype, device);
}

infinicore::Tensor Qwen3NextGatedDeltaNet::forward(const infinicore::Tensor &hidden_states) const {

    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto qkv = in_proj_qkv_->forward(hidden_states_mutable);
    auto z = in_proj_z_->forward(hidden_states_mutable);
    auto a = in_proj_a_->forward(hidden_states_mutable);
    auto b = in_proj_b_->forward(hidden_states_mutable);

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &mamba_metadata = forward_context.mamba_metadata;

    auto conv_out = infinicore::op::causal_conv1d(
        qkv,
        forward_context.conv_state_vec[layer_idx_],
        conv1d_weight_,
        std::nullopt,
        mamba_metadata.input_offsets.value(),
        mamba_metadata.init_state_indices.value(),
        mamba_metadata.final_state_indices.value());
    auto conv_qkv = infinicore::op::silu(conv_out);

    auto q = conv_qkv->narrow({{2, 0, key_dim_}});
    auto k = conv_qkv->narrow({{2, key_dim_, key_dim_}});
    auto v = conv_qkv->narrow({{2, key_dim_ * 2, value_dim_}});
    bool is_decode = mamba_metadata.input_offsets.value()->shape()[0] - 1 == seq_len;
    infinicore::Tensor delta_out;
    if (is_decode) {
        auto ssm_state = forward_context.ssm_state_vec[layer_idx_];
        auto q_delta = q->view({seq_len, 1, linear_num_key_heads_, linear_key_head_dim_});
        auto k_delta = k->view({seq_len, 1, linear_num_key_heads_, linear_key_head_dim_});
        auto v_delta = v->view({seq_len, 1, linear_num_value_heads_, linear_value_head_dim_});

        auto a_heads = a->view({seq_len, 1, linear_num_value_heads_});
        auto b_heads = b->view({seq_len, 1, linear_num_value_heads_});
        auto [g, beta] = infinicore::op::fused_gated_delta_net_gating(A_log_, a_heads, b_heads, dt_bias_);

        delta_out = infinicore::op::recurrent_gated_delta_rule_indexed(
            q_delta,
            k_delta,
            v_delta,
            g,
            beta,
            ssm_state,
            mamba_metadata.init_state_indices.value(),
            mamba_metadata.final_state_indices.value(),
            true);
        delta_out = delta_out->view({seq_len, linear_num_value_heads_, linear_value_head_dim_});
    } else {
        auto ssm_state = forward_context.ssm_state_vec[layer_idx_];
        auto q_delta = q->view({1, seq_len, linear_num_key_heads_, linear_key_head_dim_});
        auto k_delta = k->view({1, seq_len, linear_num_key_heads_, linear_key_head_dim_});
        auto v_delta = v->view({1, seq_len, linear_num_value_heads_, linear_value_head_dim_});

        auto a_heads = a->view({1, seq_len, linear_num_value_heads_});
        auto b_heads = b->view({1, seq_len, linear_num_value_heads_});
        auto [g, beta] = infinicore::op::fused_gated_delta_net_gating(A_log_, a_heads, b_heads, dt_bias_);

        delta_out = infinicore::op::chunk_gated_delta_rule(
            q_delta,
            k_delta,
            v_delta,
            g,
            beta,
            ssm_state,
            mamba_metadata.input_offsets.value(),
            mamba_metadata.init_state_indices.value(),
            mamba_metadata.final_state_indices.value(),
            true);
        delta_out = delta_out->view({seq_len, linear_num_value_heads_, linear_value_head_dim_});
    }

    auto v_norm = norm_->forward(delta_out->view({batch_size * seq_len * linear_num_value_heads_, linear_value_head_dim_}))
                      ->view({batch_size, seq_len, value_dim_});
    auto gated = infinicore::op::mul(v_norm, infinicore::op::silu(z));
    return out_proj_->forward(gated);
}

} // namespace infinilm::models::qwen3_next
