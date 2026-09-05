#include "qwen3_next_gated_deltanet.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/causal_conv1d.hpp>
#include <infinicore/ops/chunk_gated_delta_rule.hpp>
#include <infinicore/ops/fused_gated_delta_net_gating.hpp>
#include <infinicore/ops/recurrent_gated_delta_rule.hpp>
#include <infinicore/ops/silu.hpp>
#include <infinicore/ops/swiglu.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextCausalConv1D::Qwen3NextCausalConv1D(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             size_t layer_idx,
                                             const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    size_t linear_num_value_heads = model_config->get<size_t>("linear_num_value_heads");
    size_t linear_num_key_heads = model_config->get<size_t>("linear_num_key_heads");
    size_t linear_key_head_dim = model_config->get<size_t>("linear_key_head_dim");
    size_t linear_value_head_dim = model_config->get<size_t>("linear_value_head_dim");
    size_t linear_conv_kernel_dim = model_config->get<size_t>("linear_conv_kernel_dim");

    size_t key_dim = linear_key_head_dim * linear_num_key_heads;
    size_t value_dim = linear_value_head_dim * linear_num_value_heads;
    size_t conv_dim = key_dim * 2 + value_dim;

    size_t conv_state_len = linear_conv_kernel_dim > 0 ? linear_conv_kernel_dim - 1 : 0;
    weight_ = infinicore::nn::Parameter({conv_dim, 1, linear_conv_kernel_dim}, dtype, device);
    this->register_parameter("weight", weight_);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = rank_info.tp_size;
    tp_rank_ = rank_info.tp_rank;
    conv_kernel_dim_ = linear_conv_kernel_dim;
    auto tp_size = tp_size_;
    full_qk_dim_ = linear_num_key_heads * linear_key_head_dim;
    full_v_dim_ = linear_num_value_heads * linear_value_head_dim;
    local_qk_dim_ = (linear_num_key_heads >= tp_size ? linear_num_key_heads / tp_size : 1) * linear_key_head_dim;
    local_v_dim_ = (linear_num_value_heads >= tp_size ? linear_num_value_heads / tp_size : 1) * linear_value_head_dim;
    local_conv_dim_ = local_qk_dim_ * 2 + local_v_dim_;
}

void Qwen3NextCausalConv1D::process_weights_after_loading() {
    if (tp_size_ <= 1 || weight_->size(0) == local_conv_dim_) {
        return;
    }

    const size_t expected_full_conv_dim = full_qk_dim_ * 2 + full_v_dim_;
    if (weight_->size(0) != expected_full_conv_dim) {
        throw std::runtime_error("Qwen3NextCausalConv1D: unexpected conv1d weight shape for TP slicing");
    }

    auto local_weight = infinicore::Tensor::empty(
        {local_conv_dim_, 1, conv_kernel_dim_},
        weight_->dtype(),
        weight_->device());

    const size_t src_qk0_offset = tp_rank_ * local_qk_dim_;
    const size_t src_qk1_offset = full_qk_dim_ + tp_rank_ * local_qk_dim_;
    const size_t src_v_offset = 2 * full_qk_dim_ + tp_rank_ * local_v_dim_;

    const size_t dst_qk0_offset = 0;
    const size_t dst_qk1_offset = local_qk_dim_;
    const size_t dst_v_offset = 2 * local_qk_dim_;

    local_weight->narrow({{0, dst_qk0_offset, local_qk_dim_}})
        ->copy_from(weight_->narrow({{0, src_qk0_offset, local_qk_dim_}}));
    local_weight->narrow({{0, dst_qk1_offset, local_qk_dim_}})
        ->copy_from(weight_->narrow({{0, src_qk1_offset, local_qk_dim_}}));
    local_weight->narrow({{0, dst_v_offset, local_v_dim_}})
        ->copy_from(weight_->narrow({{0, src_v_offset, local_v_dim_}}));

    weight_ = infinicore::nn::Parameter(local_weight);
    parameters_["weight"] = weight_;
}

infinicore::Tensor Qwen3NextCausalConv1D::forward(const infinicore::Tensor &qkv) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &mamba_metadata = forward_context.mamba_metadata;

    auto conv_out = infinicore::op::causal_conv1d(
        qkv,
        forward_context.conv_state_vec[layer_idx_],
        weight_->narrow({{0, 0, local_conv_dim_}}), // narrow in case load is skipped
        std::nullopt,
        mamba_metadata.input_offsets.value(),
        mamba_metadata.init_state_indices.value(),
        mamba_metadata.final_state_indices.value());
    auto conv_qkv = infinicore::op::silu(conv_out);
    return conv_qkv;
}

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
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto tp_size = rank_info.tp_size;
    auto tp_rank = rank_info.tp_rank;
    local_num_value_heads_ = linear_num_value_heads / tp_size;
    local_num_key_heads_ = linear_num_key_heads / tp_size;
    key_head_dim_ = linear_key_head_dim;
    value_head_dim_ = linear_value_head_dim;
    size_t key_dim = linear_key_head_dim * linear_num_key_heads;
    size_t value_dim = linear_value_head_dim * linear_num_value_heads;
    local_key_dim_ = key_head_dim_ * local_num_key_heads_;
    local_value_dim_ = value_head_dim_ * local_num_value_heads_;

    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    conv1d_ = this->register_module<Qwen3NextCausalConv1D>("conv1d", model_config, layer_idx, device);

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    if (quantization_method->get_quant_scheme()
        == infinilm::quantization::QuantScheme::NONE) {
        in_proj_qkvz_ = std::make_shared<layers::linear::MergedColumnParallelLinear>(
            hidden_size,
            std::vector<layers::linear::MergedLinearSplit>{
                {"in_proj_q", key_dim},
                {"in_proj_k", key_dim, linear_num_key_heads},
                {"in_proj_v", value_dim, linear_num_value_heads},
                {"in_proj_z", value_dim},
            },
            register_fn,
            false,
            dtype,
            device,
            rank_info);
    } else {
        in_proj_qkv_ = std::make_shared<layers::linear::QKVParallelLinear>(
            hidden_size, linear_key_head_dim, linear_key_head_dim, linear_value_head_dim, linear_num_key_heads, linear_num_key_heads, linear_num_value_heads,
            false, false, false,
            "in_proj_q", "in_proj_k", "in_proj_v", register_fn,
            quantization_method, dtype, device, rank_info);
        in_proj_z_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>("in_proj_z", hidden_size, value_dim, false, dtype, device, tp_rank, tp_size);
    }
    in_proj_ba_ = std::make_shared<layers::linear::GateUpParallelLinear>(
        hidden_size,
        linear_num_value_heads,
        "in_proj_b",
        "in_proj_a",
        register_fn,
        nullptr,
        false,
        dtype,
        device,
        rank_info);

    INFINICORE_NN_PARAMETER_INIT(dt_bias, ({linear_num_value_heads}, dtype, device, 0, tp_rank, tp_size));
    INFINICORE_NN_PARAMETER_INIT(A_log, ({linear_num_value_heads}, dtype, device, 0, tp_rank, tp_size));

    INFINICORE_NN_MODULE_INIT(norm, linear_value_head_dim, rms_norm_eps, dtype, device);
    out_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "out_proj", value_dim, hidden_size, quantization_method,
        false, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

infinicore::Tensor Qwen3NextGatedDeltaNet::forward(const infinicore::Tensor &hidden_states) const {

    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    infinicore::Tensor qkv;
    infinicore::Tensor z;
    if (in_proj_qkvz_) {
        auto qkvz = in_proj_qkvz_->forward(hidden_states_mutable);
        const size_t output_dim = qkvz->ndim() - 1;
        const size_t qkv_size = local_key_dim_ * 2 + local_value_dim_;
        qkv = qkvz->narrow({{output_dim, 0, qkv_size}});
        z = qkvz->narrow({{output_dim, qkv_size, local_value_dim_}});
    } else {
        qkv = in_proj_qkv_->forward(hidden_states_mutable);
        z = in_proj_z_->forward(hidden_states_mutable);
    }
    auto [b, a] = in_proj_ba_->forward_split(hidden_states_mutable);

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &mamba_metadata = forward_context.mamba_metadata;

    auto conv_qkv = this->conv1d_->forward(qkv);

    auto q = conv_qkv->narrow({{2, 0, local_key_dim_}});
    auto k = conv_qkv->narrow({{2, local_key_dim_, local_key_dim_}});
    auto v = conv_qkv->narrow({{2, local_key_dim_ * 2, local_value_dim_}});
    bool is_decode = mamba_metadata.input_offsets.value()->shape()[0] - 1 == seq_len;
    infinicore::Tensor delta_out;
    if (is_decode) {
        auto ssm_state = forward_context.ssm_state_vec[layer_idx_];
        auto q_delta = q->as_strided(
            {seq_len, 1, local_num_key_heads_, key_head_dim_},
            {q->stride(1), q->stride(0), static_cast<infinicore::Stride>(key_head_dim_), 1});
        auto k_delta = k->as_strided(
            {seq_len, 1, local_num_key_heads_, key_head_dim_},
            {k->stride(1), k->stride(0), static_cast<infinicore::Stride>(key_head_dim_), 1});
        auto v_delta = v->as_strided(
            {seq_len, 1, local_num_value_heads_, value_head_dim_},
            {v->stride(1), v->stride(0), static_cast<infinicore::Stride>(value_head_dim_), 1});

        auto a_heads = a->as_strided(
            {seq_len, 1, local_num_value_heads_},
            {a->stride(1), a->stride(0), 1});
        auto b_heads = b->as_strided(
            {seq_len, 1, local_num_value_heads_},
            {b->stride(1), b->stride(0), 1});
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
        delta_out = delta_out->as_strided(
            {seq_len, local_num_value_heads_, value_head_dim_},
            {delta_out->stride(0), delta_out->stride(2), delta_out->stride(3)});
    } else {
        auto ssm_state = forward_context.ssm_state_vec[layer_idx_];
        auto q_delta = q->as_strided(
            {1, seq_len, local_num_key_heads_, key_head_dim_},
            {q->stride(0), q->stride(1), static_cast<infinicore::Stride>(key_head_dim_), 1});
        auto k_delta = k->as_strided(
            {1, seq_len, local_num_key_heads_, key_head_dim_},
            {k->stride(0), k->stride(1), static_cast<infinicore::Stride>(key_head_dim_), 1});
        auto v_delta = v->as_strided(
            {1, seq_len, local_num_value_heads_, value_head_dim_},
            {v->stride(0), v->stride(1), static_cast<infinicore::Stride>(value_head_dim_), 1});

        auto a_heads = a->as_strided(
            {1, seq_len, local_num_value_heads_},
            {a->stride(0), a->stride(1), 1});
        auto b_heads = b->as_strided(
            {1, seq_len, local_num_value_heads_},
            {b->stride(0), b->stride(1), 1});
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
        delta_out = delta_out->as_strided(
            {seq_len, local_num_value_heads_, value_head_dim_},
            {delta_out->stride(1), delta_out->stride(2), delta_out->stride(3)});
    }

    auto delta_out_2d = delta_out->as_strided(
        {batch_size * seq_len * local_num_value_heads_, value_head_dim_},
        {static_cast<infinicore::Stride>(value_head_dim_), 1});
    auto v_norm_2d = norm_->forward(delta_out_2d);
    auto v_norm = v_norm_2d->as_strided(
        {batch_size, seq_len, local_value_dim_},
        {static_cast<infinicore::Stride>(seq_len * local_value_dim_), static_cast<infinicore::Stride>(local_value_dim_), 1});
    auto gated = infinicore::op::swiglu(v_norm, z);
    return out_proj_->forward(gated);
}

void Qwen3NextGatedDeltaNet::process_weights_after_loading() {
    if (in_proj_qkvz_) {
        in_proj_qkvz_->process_weights_after_loading();
    } else {
        in_proj_qkv_->process_weights_after_loading();
    }
    in_proj_ba_->process_weights_after_loading();
}

void Qwen3NextGatedDeltaNet::reset_runtime_state() const {
    if (in_proj_qkvz_) {
        in_proj_qkvz_->reset_runtime_state();
    } else {
        in_proj_qkv_->reset_runtime_state();
    }
    in_proj_ba_->reset_runtime_state();
}

} // namespace infinilm::models::qwen3_next
