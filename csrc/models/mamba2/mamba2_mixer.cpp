#include "mamba2_mixer.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/float_power.hpp"
#include "infinicore/ops/mamba2_scan.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/sum.hpp"

namespace infinilm::models::mamba2 {

infinicore::Tensor cast_activation(const infinicore::Tensor &input, infinicore::DataType dtype) {
    if (input->dtype() == dtype) {
        return input;
    }
    auto output = infinicore::Tensor::empty(input->shape(), dtype, input->device());
    infinicore::op::cast_(output, input);
    return output;
}

Mamba2Mixer::Mamba2Mixer(std::shared_ptr<config::ModelConfig> config, size_t layer_idx,
                         const infinicore::Device &device)
    : layer_idx_(layer_idx), intermediate_(config->get<size_t>("intermediate_size")),
      heads_(config->get<size_t>("num_heads")), head_dim_(config->get<size_t>("head_dim")),
      state_size_(config->get<size_t>("state_size")), conv_dim_(intermediate_ + 2 * state_size_) {
    const auto &rank = global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = rank.tp_rank;
    tp_size_ = rank.tp_size;
    communicator_ = rank.comm;
    if (heads_ % tp_size_ != 0) {
        throw std::runtime_error("Mamba-2 heads must divide evenly across tensor-parallel ranks.");
    }
    const size_t total_heads = heads_, total_intermediate = intermediate_;
    heads_ /= tp_size_;
    intermediate_ /= tp_size_;
    conv_dim_ = intermediate_ + 2 * state_size_;
    const auto dtype = config->get_dtype();
    const auto hidden = config->get<size_t>("hidden_size");
    const auto fp32 = infinicore::DataType::F32;
    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter parameter) {
        register_parameter(name, std::move(parameter));
    };
    // Reuse asymmetric projection sharding: per-head z/x/dt, replicated B/C.
    in_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden, 2 * head_dim_ + 1, state_size_, state_size_, total_heads, 1, 1,
        false, false, false, "in_proj_zxd", "in_proj_b", "in_proj_c", register_fn,
        nullptr, dtype, device, rank);
    INFINICORE_NN_MODULE_INIT(out_proj, total_intermediate, hidden, false, dtype, device, tp_rank_, tp_size_, communicator_);
    INFINICORE_NN_MODULE_INIT(norm, total_intermediate, config->get<double>("layer_norm_epsilon"), fp32, device);
    conv1d_weight_ = infinicore::Tensor::empty({conv_dim_, 1, 4}, dtype, device);
    conv1d_bias_ = infinicore::Tensor::empty({conv_dim_}, dtype, device);
    for (size_t i = 0; i < 3; ++i) {
        const size_t start = i == 0 ? 0 : intermediate_ + (i - 1) * state_size_;
        const size_t count = i == 0 ? intermediate_ : state_size_;
        const std::string name = i == 0 ? "conv1d_x" : (i == 1 ? "conv1d_b" : "conv1d_c");
        const auto shards = i == 0 ? tp_size_ : 1;
        const auto shard_rank = i == 0 ? tp_rank_ : 0;
        register_parameter(name + "_weight", infinicore::nn::Parameter(conv1d_weight_->narrow({{0, start, count}}), 0, shard_rank, shards));
        register_parameter(name + "_bias", infinicore::nn::Parameter(conv1d_bias_->narrow({{0, start, count}}), 0, shard_rank, shards));
    }
    INFINICORE_NN_PARAMETER_INIT(A, ({total_heads}, fp32, device, 0, tp_rank_, tp_size_));
    INFINICORE_NN_PARAMETER_INIT(D, ({total_heads}, fp32, device, 0, tp_rank_, tp_size_));
    INFINICORE_NN_PARAMETER_INIT(dt_bias, ({total_heads}, fp32, device, 0, tp_rank_, tp_size_));
    if (tp_size_ > 1) {
        const float scale = 1.0f / total_intermediate;
        const float epsilon = config->get<double>("layer_norm_epsilon");
        norm_scale_ = infinicore::Tensor::empty({1, 1, 1}, fp32, device);
        norm_epsilon_ = infinicore::Tensor::empty({1, 1, 1}, fp32, device);
        infinicore::context::memcpyH2D(norm_scale_->data(), &scale, sizeof(scale), false);
        infinicore::context::memcpyH2D(norm_epsilon_->data(), &epsilon, sizeof(epsilon), false);
    }
}

infinicore::Tensor Mamba2Mixer::forward(infinicore::Tensor input) const {
    auto &context = global_state::get_forward_context();
    const auto &metadata = context.mamba_metadata;
    const auto tokens = input->numel() / input->size(input->ndim() - 1);
    auto [zxd, projected_b, projected_c] = in_proj_->forward_split(input);
    auto per_head = zxd->contiguous()->view({1, tokens, heads_, 2 * head_dim_ + 1});
    auto gate = per_head->narrow({{3, 0, head_dim_}})->contiguous()->view({1, tokens, intermediate_});
    auto projected_x = per_head->narrow({{3, head_dim_, head_dim_}})->contiguous()->view({1, tokens, intermediate_});
    auto conv_input = infinicore::op::cat({projected_x, projected_b, projected_c}, 2);
    auto dt = per_head->narrow({{3, 2 * head_dim_, 1}})->contiguous()->view({tokens, heads_});
    auto convolved = infinicore::op::causal_conv1d(
        conv_input, context.conv_state_vec.at(layer_idx_), conv1d_weight_, conv1d_bias_,
        metadata.input_offsets, metadata.init_state_indices, metadata.final_state_indices);
    convolved = infinicore::op::silu(convolved);
    auto x = convolved->narrow({{2, 0, intermediate_}})->contiguous()->view({tokens, heads_, head_dim_});
    auto b = convolved->narrow({{2, intermediate_, state_size_}})->contiguous()->view({tokens, 1, state_size_});
    auto c = convolved->narrow({{2, intermediate_ + state_size_, state_size_}})->contiguous()->view({tokens, 1, state_size_});
    auto scan = infinicore::op::mamba2_scan(
        x, dt, b, c, A_, D_, dt_bias_, context.ssm_state_vec.at(layer_idx_),
        metadata.input_offsets.value(), metadata.init_state_indices.value(), metadata.final_state_indices.value());
    auto y = cast_activation(scan->view({1, tokens, intermediate_}), infinicore::DataType::F32);
    auto z = infinicore::op::silu(cast_activation(gate, infinicore::DataType::F32));
    auto gated = infinicore::op::mul(y, z);
    infinicore::Tensor normalized;
    if (tp_size_ == 1) {
        normalized = norm_->forward(gated);
    } else {
        auto squares = infinicore::op::mul(gated, gated);
        auto total = infinicore::op::sum(squares, {2}, true);
        infinicore::op::distributed::allreduce_(total, total, INFINICCL_SUM, communicator_);
        auto scale = norm_scale_->as_strided(total->shape(), {0, 0, 0});
        auto epsilon = norm_epsilon_->as_strided(total->shape(), {0, 0, 0});
        auto variance = infinicore::op::add(infinicore::op::mul(total, scale), epsilon);
        auto inverse_rms = infinicore::Tensor::empty(variance->shape(), variance->dtype(), variance->device());
        infinicore::op::float_power_(inverse_rms, variance, -0.5);
        auto weight = norm_->weight()->narrow({{0, tp_rank_ * intermediate_, intermediate_}})->as_strided(gated->shape(), {0, 0, 1});
        auto factors = inverse_rms->as_strided(gated->shape(), {0, 1, 0});
        normalized = infinicore::op::mul(infinicore::op::mul(gated, factors), weight);
    }
    normalized = cast_activation(normalized, input->dtype());
    return out_proj_->forward(normalized);
}

Mamba2Block::Mamba2Block(std::shared_ptr<config::ModelConfig> config, size_t layer_idx,
                         const infinicore::Device &device)
    : dtype_(config->get_dtype()) {
    INFINICORE_NN_MODULE_INIT(norm, config->get<size_t>("hidden_size"), config->get<double>("layer_norm_epsilon"), infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(mixer, config, layer_idx, device);
}

infinicore::Tensor Mamba2Block::forward(const infinicore::Tensor &residual) const {
    auto input = cast_activation(norm_->forward(residual), dtype_);
    auto output = cast_activation(mixer_->forward(input), infinicore::DataType::F32);
    return infinicore::op::add(residual, output);
}

} // namespace infinilm::models::mamba2
