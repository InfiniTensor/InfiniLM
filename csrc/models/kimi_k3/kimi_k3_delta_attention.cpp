#include "kimi_k3_delta_attention.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/causal_conv1d.hpp>
#include <infinicore/ops/kimi_delta_attention.hpp>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <infinicore/ops/silu.hpp>

#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::kimi_k3 {

KimiK3ShortConv::KimiK3ShortConv(size_t full_channels,
                                 size_t kernel_size,
                                 const infinicore::DataType &dtype,
                                 const infinicore::Device &device) {
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    if (full_channels % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error("KimiK3ShortConv: channels must be divisible by tp_size");
    }
    weight_ = infinicore::nn::Parameter(
        {full_channels, 1, kernel_size},
        dtype,
        device,
        0,
        rank_info.tp_rank,
        rank_info.tp_size);
    this->register_parameter("weight", weight_);
}

KimiK3DeltaAttention::KimiK3DeltaAttention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto &linear_config = model_config->get_config_json().at("linear_attn_config");
    const size_t num_heads = linear_config.at("num_heads").get<size_t>();
    head_dim_ = linear_config.at("head_dim").get<size_t>();
    conv_kernel_size_ = linear_config.at("short_conv_kernel_size").get<size_t>();
    gate_lower_bound_ = linear_config.value("gate_lower_bound", -5.0f);
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    if (num_heads % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error("KimiK3DeltaAttention: num_heads must be divisible by tp_size");
    }
    local_num_heads_ = num_heads / static_cast<size_t>(rank_info.tp_size);
    local_projection_size_ = local_num_heads_ * head_dim_;
    const size_t projection_size = num_heads * head_dim_;

    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
        hidden_size,
        head_dim_,
        num_heads,
        num_heads,
        "q_proj",
        "k_proj",
        "v_proj",
        register_fn,
        nullptr,
        false,
        dtype,
        device,
        rank_info);
    INFINICORE_NN_MODULE_INIT(q_conv1d, projection_size, conv_kernel_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_conv1d, projection_size, conv_kernel_size_, dtype, device);
    INFINICORE_NN_MODULE_INIT(v_conv1d, projection_size, conv_kernel_size_, dtype, device);
    packed_conv_weight_ = infinicore::Tensor::empty(
        {3 * local_projection_size_, 1, conv_kernel_size_}, dtype, device);

    INFINICORE_NN_PARAMETER_INIT(A_log,
                                 ({num_heads}, infinicore::DataType::F32, device,
                                  0, rank_info.tp_rank, rank_info.tp_size));
    INFINICORE_NN_MODULE_INIT(f_a_proj, hidden_size, head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(f_b_proj, head_dim_, projection_size, false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_PARAMETER_INIT(dt_bias,
                                 ({projection_size}, infinicore::DataType::F32, device,
                                  0, rank_info.tp_rank, rank_info.tp_size));
    INFINICORE_NN_MODULE_INIT(b_proj, hidden_size, num_heads, false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(g_proj, hidden_size, projection_size, false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(o_norm, head_dim_, model_config->get<double>("rms_norm_eps"),
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(o_proj, projection_size, hidden_size, false, dtype, device,
                              rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

void KimiK3DeltaAttention::process_weights_after_loading() {
    qkv_proj_->process_weights_after_loading();

    const std::array<infinicore::Tensor, 3> weights{
        q_conv1d_->weight(),
        k_conv1d_->weight(),
        v_conv1d_->weight(),
    };
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i]->shape() != infinicore::Shape{local_projection_size_, 1, conv_kernel_size_}) {
            throw std::runtime_error(
                "KimiK3DeltaAttention: unexpected short convolution weight shape");
        }
        packed_conv_weight_->narrow(
                               {{0, i * local_projection_size_, local_projection_size_}})
            ->copy_from(weights[i]);
    }
}

void KimiK3DeltaAttention::reset_runtime_state() const {
    qkv_proj_->reset_runtime_state();
}

infinicore::Tensor KimiK3DeltaAttention::forward(
    const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto input = hidden_states;
    auto qkv_projected = qkv_proj_->forward(input);
    auto &context = global_state::get_forward_context();
    auto &metadata = context.mamba_metadata;
    auto qkv = infinicore::op::causal_conv1d(
        qkv_projected,
        context.conv_state_vec.at(layer_idx_),
        packed_conv_weight_,
        std::nullopt,
        metadata.input_offsets.value(),
        metadata.init_state_indices.value(),
        metadata.final_state_indices.value());
    qkv = infinicore::op::silu(qkv);
    auto q = qkv->narrow({{2, 0, local_projection_size_}});
    auto k = qkv->narrow({{2, local_projection_size_, local_projection_size_}});
    auto v = qkv->narrow({{2, 2 * local_projection_size_, local_projection_size_}});
    auto f_a = f_a_proj_->forward(input);
    auto gate_decay_projected = f_b_proj_->forward(f_a);
    auto gate_decay = gate_decay_projected->view(
        {batch_size, seq_len, local_num_heads_, head_dim_});
    auto beta_projected = b_proj_->forward(input);
    auto beta = beta_projected->view(
        {batch_size, seq_len, local_num_heads_});
    auto q_heads = q->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto k_heads = k->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto v_heads = v->view({batch_size, seq_len, local_num_heads_, head_dim_});

    auto dt_bias = dt_bias_->view({local_num_heads_, head_dim_});
    auto output = infinicore::op::kimi_delta_attention(
        q_heads,
        k_heads,
        v_heads,
        gate_decay,
        beta,
        A_log_,
        dt_bias,
        context.ssm_state_vec.at(layer_idx_),
        metadata.input_offsets,
        metadata.init_state_indices,
        metadata.final_state_indices,
        1.0f / std::sqrt(static_cast<float>(head_dim_)),
        gate_lower_bound_,
        true);

    auto output_2d = output->view(
        {batch_size * seq_len * local_num_heads_, head_dim_});
    auto normalized_2d = o_norm_->forward(output_2d);
    auto normalized = normalized_2d->view(
        {batch_size, seq_len, local_projection_size_});
    auto output_gate_input = g_proj_->forward(input);
    auto output_gate = infinicore::op::sigmoid(output_gate_input);
    auto gated = infinicore::op::mul(normalized, output_gate);
    return o_proj_->forward(gated);
}

} // namespace infinilm::models::kimi_k3
