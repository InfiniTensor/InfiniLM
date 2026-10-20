#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "qwen3_next_gated_deltanet.hpp"

#include "../../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"


#include <cstdint>
#include <array>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextCausalConv1D::Qwen3NextCausalConv1D(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || !model_config || rank_info.device != runtime.device()
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Next construction requires matching eager Runtime/rank");
    (void)attention_backend; (void)ep_config;
    device_ = runtime.device();
    layer_idx_ = layer_idx;
    const auto dtype = model_config->get_dtype();
    weight_dtype_ = dtype;
    const size_t linear_num_value_heads = model_config->get<size_t>("linear_num_value_heads");
    const size_t linear_num_key_heads = model_config->get<size_t>("linear_num_key_heads");
    const size_t linear_key_head_dim = model_config->get<size_t>("linear_key_head_dim");
    const size_t linear_value_head_dim = model_config->get<size_t>("linear_value_head_dim");
    const size_t linear_conv_kernel_dim = model_config->get<size_t>("linear_conv_kernel_dim");
    const size_t key_dim = linear_key_head_dim * linear_num_key_heads;
    const size_t value_dim = linear_value_head_dim * linear_num_value_heads;
    const size_t conv_dim = key_dim * 2 + value_dim;
    tp_size_ = rank_info.tp_size;
    tp_rank_ = rank_info.tp_rank;
    conv_kernel_dim_ = linear_conv_kernel_dim;
    const auto tp_size = tp_size_;
    full_qk_dim_ = linear_num_key_heads * linear_key_head_dim;
    full_v_dim_ = linear_num_value_heads * linear_value_head_dim;
    local_qk_dim_ = (linear_num_key_heads >= tp_size ? linear_num_key_heads / tp_size : 1) * linear_key_head_dim;
    local_v_dim_ = (linear_num_value_heads >= tp_size ? linear_num_value_heads / tp_size : 1) * linear_value_head_dim;
    local_conv_dim_ = local_qk_dim_ * 2 + local_v_dim_;
    try {
        weight_ = infinilm::weights::Weight(runtime, {conv_dim, 1, linear_conv_kernel_dim}, dtype);
        this->bind_weight("weight", weight_);
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

void Qwen3NextCausalConv1D::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool /*pre_transpose*/) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_ || !weight_ || weight_->device() != device_)
        throw std::invalid_argument("Qwen3NextCausalConv1D: postload Runtime/weight device mismatch");
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("Qwen3NextCausalConv1D: postload requires inactive graph state");
    if (postload_failed_)
        throw std::logic_error("Qwen3NextCausalConv1D: failed postload cannot be retried");

    const size_t max = std::numeric_limits<size_t>::max();
    if (!tp_size_ || tp_rank_ >= tp_size_ || !conv_kernel_dim_ || !local_qk_dim_ || !local_v_dim_
        || full_qk_dim_ > (max - full_v_dim_) / 2
        || local_qk_dim_ > (max - local_v_dim_) / 2
        || full_qk_dim_ % tp_size_ || full_v_dim_ % tp_size_
        || full_qk_dim_ / tp_size_ != local_qk_dim_ || full_v_dim_ / tp_size_ != local_v_dim_
        || local_conv_dim_ != local_qk_dim_ * 2 + local_v_dim_)
        throw std::invalid_argument("Qwen3NextCausalConv1D: unsupported TP slicing geometry");

    const size_t expected_full_conv_dim = full_qk_dim_ * 2 + full_v_dim_;
    const auto &shape = weight_->shape();
    if (weight_->dtype() != weight_dtype_ || shape.size() != 3 || shape[1] != 1
        || shape[2] != conv_kernel_dim_
        || (shape[0] != expected_full_conv_dim && shape[0] != local_conv_dim_))
        throw std::invalid_argument("Qwen3NextCausalConv1D: expected [full/local channels,1,kernel] weight and model dtype");
    auto alias = local_weights_.find("weight");
    if (alias == local_weights_.end())
        throw std::logic_error("Qwen3NextCausalConv1D: missing registered weight");
    // Complete validation also applies to TP1 and an already-local weight.
    if (tp_size_ == 1 || shape[0] == local_conv_dim_) return;

    auto old_weight = weight_;
    auto old_alias = alias->second;
    auto local_weight = infinilm::runtime::Tensor::empty(runtime,
        {local_conv_dim_, 1, conv_kernel_dim_}, weight_dtype_);
    infinilm::weights::Weight pending_weight(local_weight);

    const size_t src_qk0_offset = tp_rank_ * local_qk_dim_;
    const size_t src_qk1_offset = full_qk_dim_ + tp_rank_ * local_qk_dim_;
    const size_t src_v_offset = 2 * full_qk_dim_ + tp_rank_ * local_v_dim_;

    const size_t dst_qk0_offset = 0;
    const size_t dst_qk1_offset = local_qk_dim_;
    const size_t dst_v_offset = 2 * local_qk_dim_;

    // Prepare ALL source/destination views before the first possible enqueue.
    // Their owners, old aliases and pending storage outlive success/catch drains.
    const std::array<infinilm::runtime::Tensor, 3> sources{
        old_weight->narrow({{0, src_qk0_offset, local_qk_dim_}}),
        old_weight->narrow({{0, src_qk1_offset, local_qk_dim_}}),
        old_weight->narrow({{0, src_v_offset, local_v_dim_}})};
    std::array<infinilm::runtime::Tensor, 3> destinations{
        local_weight->narrow({{0, dst_qk0_offset, local_qk_dim_}}),
        local_weight->narrow({{0, dst_qk1_offset, local_qk_dim_}}),
        local_weight->narrow({{0, dst_v_offset, local_v_dim_}})};
    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i]->shape() != destinations[i]->shape()
            || sources[i]->dtype() != destinations[i]->dtype()
            || sources[i]->device() != runtime.device() || destinations[i]->device() != runtime.device())
            throw std::invalid_argument("Qwen3NextCausalConv1D: invalid TP copy operands");
    }
    postload_failed_ = true; // A possibly submitted failed attempt is fail-stop.
    try {
        for (size_t i = 0; i < sources.size(); ++i)
            destinations[i]->copy_from(runtime, sources[i]);
        // Outer worker completion is too late for these local view owners.
        runtime.syncStream();
        weight_ = pending_weight;
        alias->second = pending_weight;
        postload_failed_ = false;
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor Qwen3NextCausalConv1D::forward(const infinilm::runtime::Tensor &qkv,
                                               const infinilm::global_state::ForwardContext &forward_context) const {
    if (!forward_context.runtime) {
        throw std::invalid_argument("Qwen3NextCausalConv1D forward requires its worker Runtime");
    }
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (runtime.device() != device_ || !qkv || qkv->device() != device_
        || !weight_ || weight_->device() != device_)
        throw std::invalid_argument("Qwen3NextCausalConv1D forward Runtime/input/weight device mismatch");
    const auto &mamba_metadata = forward_context.mamba_metadata;

    const auto &conv_state = forward_context.conv_state_vec[layer_idx_];
    auto conv_weight = weight_->narrow({{0, 0, local_conv_dim_}}); // narrow in case load is skipped
    const auto &conv_offsets = mamba_metadata.input_offsets.value();
    const auto &conv_initial = mamba_metadata.init_state_indices.value();
    const auto &conv_final = mamba_metadata.final_state_indices.value();
    if (!conv_state || qkv->ndim() != 3 || conv_state->ndim() != 3 || conv_weight->ndim() != 3
        || conv_state->device() != runtime.device())
        throw std::invalid_argument("causal conv required input shape/device mismatch");
    if (conv_weight->shape()[2] != 4)
        throw std::invalid_argument("causal conv allocating API requires qualified K4");
    if (!conv_offsets || conv_offsets->ndim() != 1 || conv_offsets->numel() < 2)
        throw std::invalid_argument("causal conv cu_seqlens shape mismatch");
    auto conv_out = infinilm::runtime::Tensor::empty(runtime, qkv->shape(), qkv->dtype());
    const std::optional<infinilm::runtime::Tensor> no_conv_tensor;
    infinilm::execution::submit<infini::ops::CausalConv1d>(
        runtime, infinilm::execution::RecordingPolicy::HostOnly,
        infini::ops::CausalConv1d::DefaultConfig(runtime.device().type()),
        qkv, conv_weight, conv_state, no_conv_tensor,
        std::optional<infinilm::runtime::Tensor>{conv_offsets}, std::optional<infinilm::runtime::Tensor>{conv_initial},
        std::optional<infinilm::runtime::Tensor>{conv_final}, conv_out, no_conv_tensor);
    INFINILM_TRACE_OP(CausalConv1d, conv_out);
    auto conv_qkv = infinilm::runtime::Tensor::empty(runtime, conv_out->shape(), conv_out->dtype());
    infinilm::execution::submit<infini::ops::Silu>(
        runtime, infini::ops::Silu::DefaultConfig(runtime.device().type()), conv_out, conv_qkv);
    return conv_qkv;
}

Qwen3NextGatedDeltaNet::Qwen3NextGatedDeltaNet(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || !model_config || rank_info.device != runtime.device()
        || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Next construction requires matching eager Runtime/rank");
    device_ = runtime.device();
    layer_idx_ = layer_idx;
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t linear_num_value_heads = model_config->get<size_t>("linear_num_value_heads");
    const size_t linear_num_key_heads = model_config->get<size_t>("linear_num_key_heads");
    const size_t linear_key_head_dim = model_config->get<size_t>("linear_key_head_dim");
    const size_t linear_value_head_dim = model_config->get<size_t>("linear_value_head_dim");
    const size_t tp_size = static_cast<size_t>(rank_info.tp_size);
    const auto tp_rank = rank_info.tp_rank;
    // These existing local-head expressions divide, rather than replicate.
    if (!linear_num_key_heads || !linear_num_value_heads || linear_num_key_heads % tp_size
        || linear_num_value_heads % tp_size || linear_num_value_heads % linear_num_key_heads)
        throw std::invalid_argument("Qwen3NextGatedDeltaNet: unsupported TP head geometry");
    local_num_value_heads_ = linear_num_value_heads / tp_size;
    local_num_key_heads_ = linear_num_key_heads / tp_size;
    key_head_dim_ = linear_key_head_dim;
    value_head_dim_ = linear_value_head_dim;
    const size_t value_dim = linear_value_head_dim * linear_num_value_heads;
    local_key_dim_ = key_head_dim_ * local_num_key_heads_;
    local_value_dim_ = value_head_dim_ * local_num_value_heads_;
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    auto quantization_method = model_config->get_quantization_method();
    try {
        conv1d_ = std::make_shared<Qwen3NextCausalConv1D>(
            runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
        this->bind_child("conv1d", conv1d_);
        auto register_fn = [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); };
    in_proj_qkv_ = std::make_shared<layers::linear::QKVParallelLinear>(
            runtime, hidden_size, linear_key_head_dim, linear_key_head_dim, linear_value_head_dim,
            linear_num_key_heads, linear_num_key_heads, linear_num_value_heads, false, false, false,
            "in_proj_q", "in_proj_k", "in_proj_v", register_fn, quantization_method, dtype, rank_info);
        in_proj_z_ = std::make_shared<layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, value_dim, false, dtype, tp_rank, tp_size);
        this->bind_child("in_proj_z", in_proj_z_);
        in_proj_a_ = std::make_shared<layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, linear_num_value_heads, false, dtype, tp_rank, tp_size);
        this->bind_child("in_proj_a", in_proj_a_);
        in_proj_b_ = std::make_shared<layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, linear_num_value_heads, false, dtype, tp_rank, tp_size);
        this->bind_child("in_proj_b", in_proj_b_);
        dt_bias_ = infinilm::weights::Weight(runtime, {linear_num_value_heads}, dtype, 0, tp_rank, tp_size);
        this->bind_weight("dt_bias", dt_bias_);
        A_log_ = infinilm::weights::Weight(runtime, {linear_num_value_heads}, dtype, 0, tp_rank, tp_size);
        this->bind_weight("A_log", A_log_);
        norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, linear_value_head_dim, rms_norm_eps, dtype);
        this->bind_child("norm", norm_);
        out_proj_ = std::make_shared<layers::linear::RowParallelLinear>(
            runtime, value_dim, hidden_size, quantization_method,
            false, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("out_proj", out_proj_);
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

void Qwen3NextGatedDeltaNet::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("Qwen3NextGatedDeltaNet: postload Runtime device mismatch");
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("Qwen3NextGatedDeltaNet: postload requires inactive graph state");
    // Only the hidden projection: conv1d/z/a/b/norm/out_proj are traversal-owned.
    auto old_parameters = in_proj_qkv_->state_dict();
    auto old_aliases = local_weights_;
    try {
        in_proj_qkv_->process_weights_after_loading(runtime, pre_transpose);
        runtime.syncStream(); // Old storage/split aliases survive successful transpose copies.
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor Qwen3NextGatedDeltaNet::forward(
    const infinilm::runtime::Tensor &hidden_states,
    const infinilm::global_state::ForwardContext &forward_context) const {

    if (!forward_context.runtime) {
        throw std::invalid_argument("Qwen3NextGatedDeltaNet forward requires its worker Runtime");
    }
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (runtime.device() != device_ || !hidden_states || hidden_states->device() != device_)
        throw std::invalid_argument("Qwen3NextGatedDeltaNet forward Runtime/input device mismatch");
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto qkv = in_proj_qkv_->forward(runtime, hidden_states_mutable);
    auto z = in_proj_z_->forward(runtime, hidden_states_mutable);
    auto a = in_proj_a_->forward(runtime, hidden_states_mutable);
    auto b = in_proj_b_->forward(runtime, hidden_states_mutable);

    const auto &mamba_metadata = forward_context.mamba_metadata;

    auto conv_qkv = this->conv1d_->forward(qkv, forward_context);

    auto q = conv_qkv->narrow({{2, 0, local_key_dim_}});
    auto k = conv_qkv->narrow({{2, local_key_dim_, local_key_dim_}});
    auto v = conv_qkv->narrow({{2, local_key_dim_ * 2, local_value_dim_}});
    bool is_decode = mamba_metadata.input_offsets.value()->shape()[0] - 1 == seq_len;
    infinilm::runtime::Tensor delta_out;
    if (is_decode) {
        auto ssm_state = forward_context.ssm_state_vec[layer_idx_];
        auto q_delta = q->as_strided(
            {seq_len, 1, local_num_key_heads_, key_head_dim_},
            {q->stride(1), q->stride(0), static_cast<infinilm::runtime::Stride>(key_head_dim_), 1});
        auto k_delta = k->as_strided(
            {seq_len, 1, local_num_key_heads_, key_head_dim_},
            {k->stride(1), k->stride(0), static_cast<infinilm::runtime::Stride>(key_head_dim_), 1});
        auto v_delta = v->as_strided(
            {seq_len, 1, local_num_value_heads_, value_head_dim_},
            {v->stride(1), v->stride(0), static_cast<infinilm::runtime::Stride>(value_head_dim_), 1});

        auto a_heads = a->as_strided(
            {seq_len, 1, local_num_value_heads_},
            {a->stride(1), a->stride(0), 1});
        auto b_heads = b->as_strided(
            {seq_len, 1, local_num_value_heads_},
            {b->stride(1), b->stride(0), 1});
        const auto &gating_A_log = A_log_.tensor();
        const auto &gating_dt_bias = dt_bias_.tensor();
        if (!a_heads || a_heads->device() != runtime.device())
            throw std::invalid_argument("gating input device mismatch");
        auto g = infinilm::runtime::Tensor::empty(runtime, a_heads->shape(), infinilm::runtime::DataType::kFloat32);
        auto beta = infinilm::runtime::Tensor::empty(runtime, a_heads->shape(), infinilm::runtime::DataType::kFloat32);
        const std::array<infinilm::runtime::Tensor, 6> gating_tensors{
            gating_A_log, a_heads, b_heads, gating_dt_bias, g, beta};
        for (const auto &gating_tensor : gating_tensors)
            if (!gating_tensor || gating_tensor->device() != runtime.device())
                throw std::invalid_argument("gating Runtime/tensor device mismatch");
#if defined(INFINILM_FUSED_GATED_DELTA_NET_GATING_OPS_UNAVAILABLE)
        throw std::runtime_error("FusedGatedDeltaNetGating public API is unavailable in the installed SDK");
#else
        infinilm::execution::submit<infini::ops::FusedGatedDeltaNetGating>(
            runtime, infinilm::execution::RecordingPolicy::HostOnly,
            infini::ops::FusedGatedDeltaNetGating::DefaultConfig(runtime.device().type()),
            gating_A_log, a_heads, b_heads, gating_dt_bias, 1.0f, 20.0f, g, beta);
#endif
        INFINILM_TRACE_OP(FusedGatedDeltaNetGating, g);

        const auto &delta_initial_indices = mamba_metadata.init_state_indices.value();
        const auto &delta_final_indices = mamba_metadata.final_state_indices.value();
        delta_out = infinilm::runtime::Tensor::empty(runtime, v_delta->shape(), v_delta->dtype());
        const std::array<infinilm::runtime::Tensor, 8> delta_inputs{
            q_delta, k_delta, v_delta, g, beta, ssm_state,
            delta_initial_indices, delta_final_indices};
        for (const auto &delta_input : delta_inputs)
            if (!delta_input || delta_input->device() != runtime.device())
                throw std::invalid_argument("recurrent delta Runtime/tensor device mismatch");
#if defined(INFINILM_RECURRENT_GATED_DELTA_RULE_OPS_UNAVAILABLE)
        throw std::runtime_error("RecurrentGatedDeltaRule public API is unavailable in the installed SDK");
#else
        infinilm::execution::submit<infini::ops::RecurrentGatedDeltaRule>(
            runtime, infinilm::execution::RecordingPolicy::HostOnly,
            infini::ops::RecurrentGatedDeltaRule::DefaultConfig(runtime.device().type()),
            q_delta, k_delta, v_delta, g, beta, ssm_state,
            std::optional<infinilm::runtime::Tensor>{delta_initial_indices},
            std::optional<infinilm::runtime::Tensor>{delta_final_indices}, delta_out,
            std::optional<infinilm::runtime::Tensor>{}, true);
#endif
        INFINILM_TRACE_OP(RecurrentGatedDeltaRule, delta_out);
        delta_out = delta_out->as_strided(
            {seq_len, local_num_value_heads_, value_head_dim_},
            {delta_out->stride(0), delta_out->stride(2), delta_out->stride(3)});
    } else {
        auto ssm_state = forward_context.ssm_state_vec[layer_idx_];
        auto q_delta = q->as_strided(
            {1, seq_len, local_num_key_heads_, key_head_dim_},
            {q->stride(0), q->stride(1), static_cast<infinilm::runtime::Stride>(key_head_dim_), 1});
        auto k_delta = k->as_strided(
            {1, seq_len, local_num_key_heads_, key_head_dim_},
            {k->stride(0), k->stride(1), static_cast<infinilm::runtime::Stride>(key_head_dim_), 1});
        auto v_delta = v->as_strided(
            {1, seq_len, local_num_value_heads_, value_head_dim_},
            {v->stride(0), v->stride(1), static_cast<infinilm::runtime::Stride>(value_head_dim_), 1});

        auto a_heads = a->as_strided(
            {1, seq_len, local_num_value_heads_},
            {a->stride(0), a->stride(1), 1});
        auto b_heads = b->as_strided(
            {1, seq_len, local_num_value_heads_},
            {b->stride(0), b->stride(1), 1});
        const auto &gating_A_log = A_log_.tensor();
        const auto &gating_dt_bias = dt_bias_.tensor();
        if (!a_heads || a_heads->device() != runtime.device())
            throw std::invalid_argument("gating input device mismatch");
        auto g = infinilm::runtime::Tensor::empty(runtime, a_heads->shape(), infinilm::runtime::DataType::kFloat32);
        auto beta = infinilm::runtime::Tensor::empty(runtime, a_heads->shape(), infinilm::runtime::DataType::kFloat32);
        const std::array<infinilm::runtime::Tensor, 6> gating_tensors{
            gating_A_log, a_heads, b_heads, gating_dt_bias, g, beta};
        for (const auto &gating_tensor : gating_tensors)
            if (!gating_tensor || gating_tensor->device() != runtime.device())
                throw std::invalid_argument("gating Runtime/tensor device mismatch");
#if defined(INFINILM_FUSED_GATED_DELTA_NET_GATING_OPS_UNAVAILABLE)
        throw std::runtime_error("FusedGatedDeltaNetGating public API is unavailable in the installed SDK");
#else
        infinilm::execution::submit<infini::ops::FusedGatedDeltaNetGating>(
            runtime, infinilm::execution::RecordingPolicy::HostOnly,
            infini::ops::FusedGatedDeltaNetGating::DefaultConfig(runtime.device().type()),
            gating_A_log, a_heads, b_heads, gating_dt_bias, 1.0f, 20.0f, g, beta);
#endif
        INFINILM_TRACE_OP(FusedGatedDeltaNetGating, g);

        const auto &chunk_offsets = mamba_metadata.input_offsets.value();
        const auto &chunk_initial = mamba_metadata.init_state_indices.value();
        const auto &chunk_final = mamba_metadata.final_state_indices.value();
        if (!q_delta || !k_delta || !v_delta || !ssm_state
            || q_delta->ndim() != 4 || k_delta->ndim() != 4 || v_delta->ndim() != 4)
            throw std::invalid_argument("chunk delta required sequence rank mismatch");
        if (!chunk_offsets || chunk_offsets->ndim() != 1 || chunk_offsets->numel() < 2)
            throw std::invalid_argument("chunk delta cu shape mismatch");
        delta_out = infinilm::runtime::Tensor::empty(runtime, v_delta->shape(), v_delta->dtype());
        for (const auto &chunk_input : {q_delta, k_delta, v_delta, g, beta, ssm_state,
                                       chunk_offsets, chunk_initial, chunk_final}) {
            if (!chunk_input || chunk_input->device() != runtime.device())
                throw std::invalid_argument("chunk delta Runtime/tensor device mismatch");
        }
        const std::optional<infinilm::runtime::Tensor> no_chunk_final_state;
#if defined(INFINILM_CHUNK_GATED_DELTA_RULE_OPS_UNAVAILABLE)
        throw std::runtime_error("ChunkGatedDeltaRule public API is unavailable in the installed SDK");
#else
        infinilm::execution::submit<infini::ops::ChunkGatedDeltaRule>(
            runtime, infinilm::execution::RecordingPolicy::HostOnly,
            infini::ops::ChunkGatedDeltaRule::DefaultConfig(runtime.device().type()),
            q_delta, k_delta, v_delta, g, beta, ssm_state,
            std::optional<infinilm::runtime::Tensor>{chunk_offsets},
            std::optional<infinilm::runtime::Tensor>{chunk_initial},
            std::optional<infinilm::runtime::Tensor>{chunk_final},
            delta_out, no_chunk_final_state, true, size_t{64});
#endif
        INFINILM_TRACE_OP(ChunkGatedDeltaRule, delta_out);
        delta_out = delta_out->as_strided(
            {seq_len, local_num_value_heads_, value_head_dim_},
            {delta_out->stride(1), delta_out->stride(2), delta_out->stride(3)});
    }

    auto delta_out_2d = delta_out->as_strided(
        {batch_size * seq_len * local_num_value_heads_, value_head_dim_},
        {static_cast<infinilm::runtime::Stride>(value_head_dim_), 1});
    auto v_norm_2d = norm_->forward(runtime, delta_out_2d);
    auto v_norm = v_norm_2d->as_strided(
        {batch_size, seq_len, local_value_dim_},
        {static_cast<infinilm::runtime::Stride>(seq_len * local_value_dim_), static_cast<infinilm::runtime::Stride>(local_value_dim_), 1});
    const auto ops_other = [&]() {
        auto gated = infinilm::runtime::Tensor::empty(runtime, z->shape(), z->dtype());
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Silu>(
            runtime, infini::ops::Silu::DefaultConfig(runtime.device().type()), z, gated);
        return gated;
    }();
    auto gated = infinilm::runtime::Tensor::empty(runtime, v_norm->shape(), v_norm->dtype());
    INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
        && runtime.device().type() != infinilm::runtime::Device::Type::kMars
        && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Mul>(
        runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), v_norm, ops_other, gated);
    INFINILM_TRACE_OP(Mul, gated);
    return out_proj_->forward(runtime, gated);
}

} // namespace infinilm::models::qwen3_next
