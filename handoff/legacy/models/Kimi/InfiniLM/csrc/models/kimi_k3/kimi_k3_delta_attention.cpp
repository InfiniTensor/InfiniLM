#include "kimi_k3_kda_trace.hpp"
#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "kimi_k3_delta_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"

#include <runtime/context.hpp>


#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::kimi_k3 {

namespace {

void checked_construction_rank(infinilm::runtime::Runtime &runtime,
                               const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("K3 construction requires inactive graph state");
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size)
        throw std::invalid_argument("K3 construction rank/device differs");
}

// Match existing Parameter physical stride/byte bounds before any child allocates.
void checked_parameter_geometry(std::initializer_list<size_t> shape,
                                const infinilm::runtime::DataType &dtype) {
    size_t elements = 1;
    const auto max_stride = static_cast<size_t>(std::numeric_limits<infinilm::runtime::Stride>::max());
    for (auto it = std::rbegin(shape); it != std::rend(shape); ++it) {
        if (!*it || elements > max_stride / *it)
            throw std::length_error("K3 construction parameter geometry overflows");
        elements *= *it;
    }
    const auto bytes = infinilm::runtime::dsize(dtype);
    if (!bytes || elements > std::numeric_limits<size_t>::max() / bytes)
        throw std::length_error("K3 construction parameter byte size overflows");
}
} // namespace

KimiK3ShortConv::KimiK3ShortConv(infinilm::runtime::Runtime &runtime,
                                 size_t full_channels,
                                 size_t kernel_size,
                                 const infinilm::runtime::DataType &dtype,
                                 const engine::distributed::RankInfo &rank_info) {
    checked_construction_rank(runtime, rank_info);
    const auto tp_size = static_cast<size_t>(rank_info.tp_size);
    if (!full_channels || !kernel_size)
        throw std::invalid_argument("KimiK3ShortConv: channels/kernel must be positive");
    if (full_channels % tp_size != 0)
        throw std::runtime_error("KimiK3ShortConv: channels must be divisible by tp_size");
    checked_parameter_geometry({full_channels / tp_size, 1, kernel_size}, dtype);
    device_ = runtime.device();
    try {
        weight_ = infinilm::weights::Weight(runtime,
            {full_channels, 1, kernel_size}, dtype,
            0, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("weight", weight_);
    } catch (...) {
        const auto failure = std::current_exception();
        // weight_ and registered aliases remain alive in this ordinary body.
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}


KimiK3DeltaAttention::KimiK3DeltaAttention(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const engine::distributed::RankInfo &rank_info)
    : layer_idx_(layer_idx) {
    checked_construction_rank(runtime, rank_info);
    if (!model_config)
        throw std::invalid_argument("KimiK3DeltaAttention: model config is null");
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto &linear_config = model_config->get_config_json().at("linear_attn_config");
    const size_t num_heads = linear_config.at("num_heads").get<size_t>();
    head_dim_ = linear_config.at("head_dim").get<size_t>();
    conv_kernel_size_ = linear_config.at("short_conv_kernel_size").get<size_t>();
    gate_lower_bound_ = linear_config.value("gate_lower_bound", -5.0f);
    const double eps = model_config->get<double>("rms_norm_eps");
    const auto tp_size = static_cast<size_t>(rank_info.tp_size);
    if (!hidden_size || !num_heads || !head_dim_ || head_dim_ > 128 || conv_kernel_size_ != 4
        || !std::isfinite(gate_lower_bound_) || gate_lower_bound_ > 0
        || !std::isfinite(eps) || eps <= 0)
        throw std::invalid_argument("KimiK3DeltaAttention: invalid construction geometry/config");
    if (num_heads % tp_size != 0)
        throw std::runtime_error("KimiK3DeltaAttention: num_heads must be divisible by tp_size");
    if (rank_info.tp_size > 1 && !rank_info.comm)
        throw std::invalid_argument("KimiK3DeltaAttention: TP communicator is missing");
    const auto max = std::numeric_limits<size_t>::max();
    if (num_heads > max / head_dim_ || num_heads * head_dim_ > max / 3)
        throw std::length_error("KimiK3DeltaAttention: projection size overflows");
    local_num_heads_ = num_heads / tp_size;
    local_projection_size_ = local_num_heads_ * head_dim_;
    const size_t projection_size = num_heads * head_dim_;
    // QKV, ShortConv, replicated/column/row Linear and scalar/norm layouts.
    // Every eventual allocation is preflighted before the first QKV child.
    checked_parameter_geometry({3 * local_projection_size_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, 1, conv_kernel_size_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({3 * local_projection_size_, 1, conv_kernel_size_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({local_num_heads_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({head_dim_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, head_dim_}, dtype);
    checked_parameter_geometry({local_projection_size_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({local_num_heads_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, hidden_size}, dtype);
    checked_parameter_geometry({head_dim_}, dtype);
    checked_parameter_geometry({hidden_size, local_projection_size_}, dtype);
    device_ = runtime.device();
    conv_dtype_ = infinilm::runtime::DataType::kFloat32;

    try {
        auto register_fn = [this](const std::string &name, infinilm::weights::Weight parameter) {
            this->bind_weight(name, std::move(parameter));
    };
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
            runtime, hidden_size, head_dim_, num_heads, num_heads,
            "q_proj", "k_proj", "v_proj", register_fn, nullptr, false, dtype, rank_info);
        // Retain each constructed child in a member BEFORE registration: an
        // bind_child allocation failure cannot destroy it before the drain.
        q_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, infinilm::runtime::DataType::kFloat32, rank_info);
        this->bind_child("q_conv1d", q_conv1d_);
        k_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, infinilm::runtime::DataType::kFloat32, rank_info);
        this->bind_child("k_conv1d", k_conv1d_);
        v_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, infinilm::runtime::DataType::kFloat32, rank_info);
        this->bind_child("v_conv1d", v_conv1d_);
        // The loaded pack is still allocated only by supplied-Runtime postload.
        A_log_ = infinilm::weights::Weight(runtime,
            {num_heads}, infinilm::runtime::DataType::kFloat32,
            0, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("A_log", A_log_);
        f_a_proj_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, hidden_size, head_dim_, false, dtype);
        this->bind_child("f_a_proj", f_a_proj_);
        f_b_proj_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, head_dim_, projection_size, false, dtype,
                              rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("f_b_proj", f_b_proj_);
        dt_bias_ = infinilm::weights::Weight(runtime,
            {projection_size}, infinilm::runtime::DataType::kFloat32,
            0, rank_info.tp_rank, rank_info.tp_size);
        this->bind_weight("dt_bias", dt_bias_);
        b_proj_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, num_heads, false, dtype,
                              rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("b_proj", b_proj_);
        g_proj_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, projection_size, false, dtype,
                              rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("g_proj", g_proj_);
        o_norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, eps, infinilm::runtime::DataType::kFloat32);
        this->bind_child("o_norm", o_norm_);
        o_proj_ = std::make_shared<infinilm::layers::linear::RowParallelLinear>(
            runtime, projection_size, hidden_size, false, dtype,
                              rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);
    } catch (...) {
        const auto failure = std::current_exception();
        // All constructed members/registered aliases are still alive here.
        // Leaf constructor failures keep their existing same-owner contract.
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

void KimiK3DeltaAttention::process_weights_after_loading(
    infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("KimiK3DeltaAttention: postload Runtime device differs");
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("KimiK3DeltaAttention: postload requires inactive graph state");
    if (postload_attempted_)
        throw std::logic_error("KimiK3DeltaAttention: postload is single-use after mutation attempt");
    if (!local_projection_size_ || !conv_kernel_size_
        || local_projection_size_ > std::numeric_limits<size_t>::max() / 3)
        throw std::invalid_argument("KimiK3DeltaAttention: invalid convolution pack geometry");

    const std::array<infinilm::runtime::Tensor, 3> weights{
        q_conv1d_->weight(),
        k_conv1d_->weight(),
        v_conv1d_->weight(),
    };
    // All three operands are validated before hidden QKV mutation or allocation.
    for (const auto &weight : weights) {
        if (!weight || weight->shape() != infinilm::runtime::Shape{local_projection_size_, 1, conv_kernel_size_}) {
            throw std::runtime_error(
                "KimiK3DeltaAttention: unexpected short convolution weight shape");
        }
        if (weight->device() != runtime.device() || weight->dtype() != conv_dtype_)
            throw std::invalid_argument("KimiK3DeltaAttention: convolution weight device/dtype differs");
    }
    // BaseLinear/FusedLinear catch-side drains protect their local publication.
    // Retain the pre-transform bindings also across their successful return:
    // eager copies do not capture Tensor owners and the worker fence is later.
    auto old_qkv_parameters = qkv_proj_->state_dict();
    auto old_qkv_quantization = qkv_proj_->get_quantization();
    auto old_pack = packed_conv_weight_;
    infinilm::runtime::Tensor pending;
    std::array<infinilm::runtime::Tensor, 3> slices;
    postload_attempted_ = true;
    try {
        // Only this hidden object is owner-finalized; registered children have
        // already been finalized by the children-first model traversal.
        qkv_proj_->process_weights_after_loading(runtime, pre_transpose);
        pending = infinilm::runtime::Tensor::empty(
            runtime, {3 * local_projection_size_, 1, conv_kernel_size_}, conv_dtype_);
        for (size_t i = 0; i < slices.size(); ++i)
            slices[i] = pending->narrow({{0, i * local_projection_size_, local_projection_size_}});
        for (size_t i = 0; i < weights.size(); ++i)
            slices[i]->copy_from(runtime, weights[i]);
        // Finite same-owner stream fence, once at this postload boundary.
        // Sources, old/new QKV bindings, old pack, pack and all slices live here.
        runtime.syncStream();
        packed_conv_weight_ = std::move(pending);
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        // Preserve the first error. Hidden aliases may already be transformed;
        // the worker must remain Failed and this owner cannot be retried.
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor KimiK3DeltaAttention::forward(
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context) const {
    forward_context.require_runtime().assertOwnerThread();
    auto &runtime = forward_context.require_runtime();
    KdaTraceBatch raw_trace(runtime,forward_context,layer_idx_);
    if (runtime.device().type() != infinilm::runtime::Device::Type::kHygon)
        throw std::runtime_error("KimiK3 indexed KDA provider is currently qualified for Hygon only");
    if (!hidden_states || hidden_states->ndim() != 3 || hidden_states->size(0) != 1
        || hidden_states->device() != runtime.device())
        throw std::invalid_argument("Kimi KDA requires same-owner packed [1,tokens,hidden] input");
    const auto &conv_preflight = forward_context.conv_state_vec.at(layer_idx_);
    const auto &state_preflight = forward_context.ssm_state_vec.at(layer_idx_);
    if (!conv_preflight || !state_preflight || state_preflight->ndim() != 4)
        throw std::invalid_argument("Kimi KDA state pools are missing");
    const size_t pool = state_preflight->size(0);
    if (!pool || state_preflight->shape() != infinilm::runtime::Shape{pool,local_num_heads_,head_dim_,head_dim_}
        || conv_preflight->shape() != infinilm::runtime::Shape{pool,3*local_projection_size_,3}
        || state_preflight->dtype() != hidden_states->dtype() || conv_preflight->dtype() != hidden_states->dtype()
        || state_preflight->device() != runtime.device() || conv_preflight->device() != runtime.device())
        throw std::invalid_argument("Kimi KDA state pool geometry/dtype/device differs");
    const auto &preflight = forward_context.mamba_metadata;
    if (!preflight.input_offsets || !preflight.init_state_indices || !preflight.final_state_indices
        || !preflight.input_offsets.value() || !preflight.init_state_indices.value() || !preflight.final_state_indices.value())
        throw std::invalid_argument("Kimi KDA requires explicit request offsets and state indices");
    const auto &metadata = forward_context.mamba_metadata;
    raw_trace.capture(hidden_states,"input");raw_trace.capture(conv_preflight,"conv_before");raw_trace.capture(state_preflight,"ssm_before");
    raw_trace.capture(metadata.input_offsets.value(),"offsets");raw_trace.capture(metadata.init_state_indices.value(),"read_rows");raw_trace.capture(metadata.final_state_indices.value(),"write_rows");
    raw_trace.capture(A_log_.tensor(),"a_log");raw_trace.capture(dt_bias_.tensor(),"dt_bias");
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto input = hidden_states;
    auto qkv_projected = qkv_proj_->forward(*forward_context.runtime, input);
    raw_trace.capture(qkv_projected,"qkv_projected");


    const auto &conv_state = forward_context.conv_state_vec.at(layer_idx_);
    if (!qkv_projected || !conv_state || !packed_conv_weight_
        || qkv_projected->ndim() != 3 || conv_state->ndim() != 3 || packed_conv_weight_->ndim() != 3
        || qkv_projected->device() != forward_context.runtime->device()
        || conv_state->device() != forward_context.runtime->device()
        || packed_conv_weight_->device() != forward_context.runtime->device())
        throw std::invalid_argument("causal conv required input shape/device mismatch");
    if (packed_conv_weight_->shape()[2] != 4)
        throw std::invalid_argument("causal conv allocating API requires qualified K4");
    if (!metadata.input_offsets.value() || metadata.input_offsets.value()->ndim() != 1 || metadata.input_offsets.value()->numel() < 2)
        throw std::invalid_argument("causal conv cu_seqlens shape mismatch");
    auto qkv = infinilm::runtime::Tensor::empty(*forward_context.runtime, qkv_projected->shape(), qkv_projected->dtype());
    const std::optional<infinilm::runtime::Tensor> no_conv_tensor;
    auto conv_config = infini::ops::CausalConv1d::DefaultConfig(forward_context.runtime->device().type());
    conv_config.set_implementation_index(0);
    infinilm::execution::submit<infini::ops::CausalConv1d>(
        *forward_context.runtime, infinilm::execution::RecordingPolicy::HostOnly,
        conv_config,
        qkv_projected, packed_conv_weight_, conv_state, no_conv_tensor,
        std::optional<infinilm::runtime::Tensor>{metadata.input_offsets.value()}, std::optional<infinilm::runtime::Tensor>{metadata.init_state_indices.value()},
        std::optional<infinilm::runtime::Tensor>{metadata.final_state_indices.value()}, qkv, no_conv_tensor);
    INFINILM_TRACE_OP(CausalConv1d, qkv);
    raw_trace.capture(qkv,"conv_output");
    auto silu_out=infinilm::runtime::Tensor::empty(*forward_context.runtime,qkv->shape(),qkv->dtype());
    INFINILM_ASSERT((*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kCpu
        && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kMars
        && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Silu>(
        (*forward_context.runtime), infini::ops::Silu::DefaultConfig((*forward_context.runtime).device().type()), qkv, silu_out);qkv=std::move(silu_out);
    raw_trace.capture(qkv,"silu_qkv");
    auto q = qkv->narrow({{2, 0, local_projection_size_}});
    auto k = qkv->narrow({{2, local_projection_size_, local_projection_size_}});
    auto v = qkv->narrow({{2, 2 * local_projection_size_, local_projection_size_}});
    auto f_a = f_a_proj_->forward(*forward_context.runtime, input);
    raw_trace.capture(f_a,"f_a");
    auto gate_decay_projected = f_b_proj_->forward(*forward_context.runtime, f_a);
    raw_trace.capture(gate_decay_projected,"gate_projected");
    auto gate_decay = gate_decay_projected->view(
        {batch_size, seq_len, local_num_heads_, head_dim_});
    auto beta_projected = b_proj_->forward(*forward_context.runtime, input);
    raw_trace.capture(beta_projected,"beta_projected");
    auto beta = beta_projected->view(
        {batch_size, seq_len, local_num_heads_});
    auto q_heads = q->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto k_heads = k->view({batch_size, seq_len, local_num_heads_, head_dim_});
    auto v_heads = v->view({batch_size, seq_len, local_num_heads_, head_dim_});

    auto dt_bias = dt_bias_->view({local_num_heads_, head_dim_});
    auto output = infinilm::runtime::Tensor::empty(
        *forward_context.runtime, q_heads->shape(), q_heads->dtype());
    const auto packed_shape = infinilm::runtime::Shape{1, batch_size * seq_len, local_num_heads_, head_dim_};
    auto kda_config = infini::ops::IndexedKimiDeltaAttention::DefaultConfig(forward_context.runtime->device().type());
    kda_config.set_implementation_index(0);
    infinilm::execution::submit<infini::ops::IndexedKimiDeltaAttention>(
        *forward_context.runtime, infinilm::execution::RecordingPolicy::HostOnly, kda_config,
        output->view(packed_shape), forward_context.ssm_state_vec.at(layer_idx_),
        q_heads->view(packed_shape), k_heads->view(packed_shape), v_heads->view(packed_shape),
        gate_decay->view(packed_shape), beta->view({1, batch_size * seq_len, local_num_heads_}),
        A_log_.tensor(), dt_bias,
        metadata.input_offsets.value(), metadata.init_state_indices.value(), metadata.final_state_indices.value(),
        1.0f / std::sqrt(static_cast<float>(head_dim_)), gate_lower_bound_, true);

    raw_trace.capture(output,"kda_output");raw_trace.capture(conv_state,"conv_after");raw_trace.capture(forward_context.ssm_state_vec.at(layer_idx_),"ssm_after");
    auto output_2d = output->view(
        {batch_size * seq_len * local_num_heads_, head_dim_});
    auto normalized_2d = o_norm_->forward(*forward_context.runtime, output_2d);
    raw_trace.capture(normalized_2d,"normalized_f32");
    auto normalized = normalized_2d->view(
        {batch_size, seq_len, local_projection_size_});
    auto output_gate_input = g_proj_->forward(*forward_context.runtime, input);
    raw_trace.capture(output_gate_input,"output_gate_input");
    auto output_gate = infinilm::runtime::Tensor::empty((*forward_context.runtime), output_gate_input->shape(), output_gate_input->dtype());
    INFINILM_ASSERT((*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kCpu
        && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kMars
        && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Sigmoid>(
        (*forward_context.runtime), infini::ops::Sigmoid::DefaultConfig((*forward_context.runtime).device().type()), output_gate_input, output_gate);
    INFINILM_TRACE_OP(Sigmoid, output_gate);
    raw_trace.capture(output_gate,"gate_sigmoid");
    auto gated=infinilm::runtime::Tensor::empty(*forward_context.runtime,normalized->shape(),normalized->dtype());
    INFINILM_ASSERT((*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kCpu
        && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kMars
        && (*forward_context.runtime).device().type() != infinilm::runtime::Device::Type::kThead);
    infinilm::execution::submit<infini::ops::Mul>(
        (*forward_context.runtime), infini::ops::Mul::DefaultConfig((*forward_context.runtime).device().type()), normalized, output_gate, gated);
    INFINILM_TRACE_OP(Mul, gated);
    raw_trace.capture(gated,"gated");
    auto result=o_proj_->forward(*forward_context.runtime,gated);
    raw_trace.capture(result,"output");raw_trace.finish();
    return result;
}

} // namespace infinilm::models::kimi_k3
