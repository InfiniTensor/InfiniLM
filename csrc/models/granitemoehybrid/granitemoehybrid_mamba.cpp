#include "granitemoehybrid_mamba.hpp"

#include "../../global_state/global_state.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/causal_conv1d.hpp"
#include "infinicore/ops/distributed/allgather.hpp"
#include "infinicore/ops/mamba_selective_scan.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/rms_norm.hpp"
#include "infinicore/ops/silu.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::granitemoehybrid {

GraniteMoeHybridCausalConv1d::GraniteMoeHybridCausalConv1d(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t mamba_expand = model_config->get_or<size_t>("mamba_expand", 2);
    const size_t mamba_n_groups = model_config->get_or<size_t>("mamba_n_groups", 1);
    const size_t mamba_n_heads = model_config->get<size_t>("mamba_n_heads");
    const size_t mamba_d_state = model_config->get<size_t>("mamba_d_state");
    const size_t mamba_d_conv = model_config->get<size_t>("mamba_d_conv");
    const size_t intermediate_size = mamba_expand * hidden_size;
    const size_t conv_dim = intermediate_size + 2 * mamba_n_groups * mamba_d_state;
    use_bias_ = model_config->get_or<bool>("mamba_conv_bias", true);

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_size_ = rank_info.tp_size;
    tp_rank_ = rank_info.tp_rank;
    conv_kernel_dim_ = mamba_d_conv;
    full_x_dim_ = intermediate_size;
    full_bc_dim_ = mamba_n_groups * mamba_d_state;
    local_x_dim_ = full_x_dim_ / tp_size_;
    local_bc_dim_ = (mamba_n_groups >= tp_size_ ? mamba_n_groups / tp_size_ : 1) * mamba_d_state;
    local_conv_dim_ = local_x_dim_ + 2 * local_bc_dim_;
    bc_replicas_ = mamba_n_groups < tp_size_ ? tp_size_ / mamba_n_groups : 1;

    INFINICORE_NN_PARAMETER_INIT(weight, ({conv_dim, 1, mamba_d_conv}, dtype, device));
    if (use_bias_) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({conv_dim}, dtype, device));
    }
}

void GraniteMoeHybridCausalConv1d::process_weights_after_loading() {
    if (tp_size_ <= 1) {
        return;
    }

    const size_t expected_full_conv_dim = full_x_dim_ + 2 * full_bc_dim_;
    const size_t bc_offset = (tp_rank_ / bc_replicas_) * local_bc_dim_;
    const size_t src_x_offset = tp_rank_ * local_x_dim_;
    const size_t src_b_offset = full_x_dim_ + bc_offset;
    const size_t src_c_offset = full_x_dim_ + full_bc_dim_ + bc_offset;

    const size_t dst_x_offset = 0;
    const size_t dst_b_offset = local_x_dim_;
    const size_t dst_c_offset = local_x_dim_ + local_bc_dim_;

    if (weight_->size(0) != local_conv_dim_) {
        if (weight_->shape() != infinicore::Shape{expected_full_conv_dim, 1, conv_kernel_dim_}) {
            throw std::runtime_error("GraniteMoeHybridCausalConv1d: unexpected conv1d weight shape for TP slicing");
        }

        auto local_weight = infinicore::Tensor::empty(
            {local_conv_dim_, 1, conv_kernel_dim_},
            weight_->dtype(),
            weight_->device());

        local_weight->narrow({{0, dst_x_offset, local_x_dim_}})
            ->copy_from(weight_->narrow({{0, src_x_offset, local_x_dim_}}));
        local_weight->narrow({{0, dst_b_offset, local_bc_dim_}})
            ->copy_from(weight_->narrow({{0, src_b_offset, local_bc_dim_}}));
        local_weight->narrow({{0, dst_c_offset, local_bc_dim_}})
            ->copy_from(weight_->narrow({{0, src_c_offset, local_bc_dim_}}));

        weight_ = infinicore::nn::Parameter(local_weight);
        parameters_["weight"] = weight_;
    }

    if (use_bias_ && bias_->size(0) != local_conv_dim_) {
        if (bias_->shape() != infinicore::Shape{expected_full_conv_dim}) {
            throw std::runtime_error("GraniteMoeHybridCausalConv1d: unexpected conv1d bias shape for TP slicing");
        }

        auto local_bias = infinicore::Tensor::empty(
            {local_conv_dim_},
            bias_->dtype(),
            bias_->device());

        local_bias->narrow({{0, dst_x_offset, local_x_dim_}})
            ->copy_from(bias_->narrow({{0, src_x_offset, local_x_dim_}}));
        local_bias->narrow({{0, dst_b_offset, local_bc_dim_}})
            ->copy_from(bias_->narrow({{0, src_b_offset, local_bc_dim_}}));
        local_bias->narrow({{0, dst_c_offset, local_bc_dim_}})
            ->copy_from(bias_->narrow({{0, src_c_offset, local_bc_dim_}}));

        bias_ = infinicore::nn::Parameter(local_bias);
        parameters_["bias"] = bias_;
    }
}

infinicore::Tensor GraniteMoeHybridCausalConv1d::forward(
    const infinicore::Tensor &input) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &mamba_metadata = forward_context.mamba_metadata;

    std::optional<infinicore::Tensor> bias = std::nullopt;
    if (use_bias_) {
        bias = bias_->narrow({{0, 0, local_conv_dim_}});
    }

    auto conv_out = infinicore::op::causal_conv1d(
        input,
        forward_context.conv_state_vec[layer_idx_],
        weight_->narrow({{0, 0, local_conv_dim_}}),
        bias,
        mamba_metadata.input_offsets,
        mamba_metadata.init_state_indices,
        mamba_metadata.final_state_indices);
    return infinicore::op::silu(conv_out);
}

GraniteMoeHybridRMSNormGated::GraniteMoeHybridRMSNormGated(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t mamba_expand = model_config->get_or<size_t>("mamba_expand", 2);
    const size_t intermediate_size = mamba_expand * hidden_size;
    eps_ = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_PARAMETER_INIT(weight, ({intermediate_size}, dtype, device));
}

infinicore::Tensor GraniteMoeHybridRMSNormGated::forward(
    const infinicore::Tensor &hidden_states,
    std::optional<infinicore::Tensor> gate) const {
    auto input = hidden_states;
    if (gate.has_value()) {
        input = infinicore::op::mul(input, infinicore::op::silu(gate.value()));
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const size_t local_size = input->size(2);
    if (rank_info.tp_size > 1) {
        input = infinicore::op::distributed::allgather(input->contiguous(), rank_info.tp_size, rank_info.comm)
                    ->view({static_cast<size_t>(rank_info.tp_size), input->size(0), input->size(1), local_size})
                    ->permute({1, 2, 0, 3})
                    ->contiguous()
                    ->view({input->size(0), input->size(1), local_size * rank_info.tp_size});
    }
    auto output = infinicore::op::rms_norm(input, weight_, static_cast<float>(eps_));
    if (rank_info.tp_size > 1) {
        output = output->narrow({{2, rank_info.tp_rank * local_size, local_size}})->contiguous();
    }
    return output;
}

GraniteMoeHybridMamba::GraniteMoeHybridMamba(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t mamba_n_heads = model_config->get<size_t>("mamba_n_heads");
    const size_t mamba_n_groups = model_config->get_or<size_t>("mamba_n_groups", 1);
    const size_t mamba_d_head = model_config->get<size_t>("mamba_d_head");
    const size_t mamba_d_state = model_config->get<size_t>("mamba_d_state");
    const size_t mamba_expand = model_config->get_or<size_t>("mamba_expand", 2);
    const size_t intermediate_size = mamba_expand * hidden_size;
    const bool mamba_proj_bias = model_config->get_or<bool>("mamba_proj_bias", false);
    if (intermediate_size != mamba_n_heads * mamba_d_head) {
        throw std::runtime_error("GraniteMoeHybridMamba: intermediate_size must equal mamba_n_heads * mamba_d_head");
    }

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    intermediate_size_ = intermediate_size / rank_info.tp_size;
    num_heads_ = mamba_n_heads / rank_info.tp_size;
    num_groups_ = mamba_n_groups >= static_cast<size_t>(rank_info.tp_size) ? mamba_n_groups / rank_info.tp_size : 1;
    head_dim_ = mamba_d_head;
    state_size_ = mamba_d_state;
    const size_t local_bc_size = num_groups_ * state_size_;
    conv_dim_ = intermediate_size_ + 2 * local_bc_size;
    const size_t local_projection_size = intermediate_size_ + conv_dim_ + num_heads_;

    in_proj_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
        hidden_size, local_projection_size * rank_info.tp_size,
        model_config->get_quantization_method(), mamba_proj_bias, dtype, device,
        rank_info.tp_rank, rank_info.tp_size);
    const std::vector<infinilm::quantization::SplitInfo> splits = {
        {"in_proj.gate", 0, intermediate_size_},
        {"in_proj.x", intermediate_size_, intermediate_size_},
        {"in_proj.B", 2 * intermediate_size_, local_bc_size, mamba_n_groups},
        {"in_proj.C", 2 * intermediate_size_ + local_bc_size, local_bc_size, mamba_n_groups},
        {"in_proj.dt", intermediate_size_ + conv_dim_, num_heads_},
    };
    for (auto &param : in_proj_->split_params(splits, rank_info.tp_rank, rank_info.tp_size, mamba_n_groups)) {
        this->register_parameter(param.full_name, std::move(param.param));
    }

    INFINICORE_NN_MODULE_INIT(conv1d, model_config, layer_idx, device);
    INFINICORE_NN_PARAMETER_INIT(dt_bias, ({mamba_n_heads}, dtype, device, 0, rank_info.tp_rank, rank_info.tp_size));
    INFINICORE_NN_PARAMETER_INIT(A_log, ({mamba_n_heads}, dtype, device, 0, rank_info.tp_rank, rank_info.tp_size));
    INFINICORE_NN_PARAMETER_INIT(D, ({mamba_n_heads}, dtype, device, 0, rank_info.tp_rank, rank_info.tp_size));
    INFINICORE_NN_MODULE_INIT(norm, model_config, device);
    INFINICORE_NN_MODULE_INIT(
        out_proj, intermediate_size, hidden_size,
        model_config->get_quantization_method(), false, dtype, device,
        rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
    if (mamba_proj_bias) {
        out_proj_bias_ = infinicore::nn::Parameter({hidden_size}, dtype, device);
        this->register_parameter("out_proj.bias", out_proj_bias_);
    }
    out_proj_->set_alpha(model_config->get_or<float>("residual_multiplier", 1.0f));
}

infinicore::Tensor GraniteMoeHybridMamba::forward(
    const infinicore::Tensor &hidden_states) const {

    const auto &hidden_shape = hidden_states->shape();
    const size_t batch_size = hidden_shape[0];
    const size_t seq_len = hidden_shape[1];
    auto projected_input = hidden_states;
    auto projected_states = in_proj_->forward(projected_input);

    auto gate = projected_states->narrow({{2, 0, intermediate_size_}})->contiguous();
    auto conv_input = projected_states->narrow(
        {{2, intermediate_size_, conv_dim_}});
    auto dt = projected_states->narrow(
        {{2, intermediate_size_ + conv_dim_, num_heads_}});

    auto conv_output = conv1d_->forward(conv_input);

    auto x = conv_output->narrow({{2, 0, intermediate_size_}})->contiguous();
    auto b = conv_output->narrow({{2, intermediate_size_, num_groups_ * state_size_}})->contiguous();
    auto c = conv_output->narrow({{2, intermediate_size_ + num_groups_ * state_size_, num_groups_ * state_size_}})->contiguous();

    dt = infinicore::op::broadcast_to(
             dt->view({batch_size, seq_len, num_heads_, 1}),
             {static_cast<int64_t>(batch_size),
              static_cast<int64_t>(seq_len),
              static_cast<int64_t>(num_heads_),
              static_cast<int64_t>(head_dim_)})
             ->view({batch_size, seq_len, intermediate_size_});

    auto a_log = infinicore::op::broadcast_to(
                     A_log_->view({num_heads_, 1, 1}),
                     {static_cast<int64_t>(num_heads_),
                      static_cast<int64_t>(head_dim_),
                      static_cast<int64_t>(state_size_)})
                     ->view({intermediate_size_, state_size_});
    auto d = infinicore::op::broadcast_to(
                 D_->view({num_heads_, 1}),
                 {static_cast<int64_t>(num_heads_), static_cast<int64_t>(head_dim_)})
                 ->view({intermediate_size_});
    auto dt_bias = infinicore::op::broadcast_to(
                       dt_bias_->view({num_heads_, 1}),
                       {static_cast<int64_t>(num_heads_), static_cast<int64_t>(head_dim_)})
                       ->view({intermediate_size_});

    const infinicore::Shape ssm_state_shape{batch_size, intermediate_size_, state_size_};
    if (!ssm_state_ || ssm_state_->shape() != ssm_state_shape) {
        ssm_state_ = infinicore::Tensor::zeros(
            ssm_state_shape,
            infinicore::DataType::F32,
            hidden_states->device());
    }

    auto scan_output = infinicore::Tensor::empty(x->shape(), x->dtype(), x->device());
    const size_t group_size = intermediate_size_ / num_groups_;
    for (size_t group = 0; group < num_groups_; ++group) {
        auto group_state = ssm_state_->narrow({{1, group * group_size, group_size}})->contiguous();
        auto group_output = infinicore::op::mamba_selective_scan(
            x->narrow({{2, group * group_size, group_size}})->contiguous(),
            dt->narrow({{2, group * group_size, group_size}})->contiguous(),
            b->narrow({{2, group * state_size_, state_size_}})->contiguous(),
            c->narrow({{2, group * state_size_, state_size_}})->contiguous(),
            a_log->narrow({{0, group * group_size, group_size}})->contiguous(),
            d->narrow({{0, group * group_size, group_size}})->contiguous(),
            gate->narrow({{2, group * group_size, group_size}})->contiguous(),
            dt_bias->narrow({{0, group * group_size, group_size}})->contiguous(),
            group_state);
        scan_output->narrow({{2, group * group_size, group_size}})->copy_from(group_output);
        if (num_groups_ > 1) {
            ssm_state_->narrow({{1, group * group_size, group_size}})->copy_from(group_state);
        }
    }

    auto normalized = norm_->forward(scan_output);
    auto output = out_proj_->forward(normalized);
    if (out_proj_bias_) {
        infinicore::op::add_(output, output, out_proj_bias_->as_strided(output->shape(), {0, 0, 1}));
    }
    return output;
}

} // namespace infinilm::models::granitemoehybrid
