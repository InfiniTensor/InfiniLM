#include "kimi_k3_delta_attention.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "execution/submit.hpp"

#include "../../global_state/global_state.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"

#include <array>
#include <cmath>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <runtime/context.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::kimi_k3 {

namespace {

void checked_construction_rank(infinilm::runtime::Runtime &runtime,
                               const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("K3 construction requires inactive graph state");
    }
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size) {
        throw std::invalid_argument("K3 construction rank/device differs");
    }
}

// Match existing Parameter physical stride/byte bounds before any child allocates.
void checked_parameter_geometry(std::initializer_list<size_t> shape,
                                const infinilm::runtime::DataType &dtype) {
    size_t elements = 1;
    const auto max_stride = static_cast<size_t>(std::numeric_limits<infinilm::runtime::Stride>::max());
    for (auto it = std::rbegin(shape); it != std::rend(shape); ++it) {
        if (!*it || elements > max_stride / *it) {
            throw std::length_error("K3 construction parameter geometry overflows");
        }
        elements *= *it;
    }
    const auto bytes = infinilm::runtime::dsize(dtype);
    if (!bytes || elements > std::numeric_limits<size_t>::max() / bytes) {
        throw std::length_error("K3 construction parameter byte size overflows");
    }
}
} // namespace

KimiK3ShortConv::KimiK3ShortConv(infinilm::runtime::Runtime &runtime,
                                 size_t full_channels,
                                 size_t kernel_size,
                                 const infinilm::runtime::DataType &dtype,
                                 const engine::distributed::RankInfo &rank_info) {
    checked_construction_rank(runtime, rank_info);
    const auto tp_size = static_cast<size_t>(rank_info.tp_size);
    if (!full_channels || !kernel_size) {
        throw std::invalid_argument("KimiK3ShortConv: channels/kernel must be positive");
    }
    if (full_channels % tp_size != 0) {
        throw std::runtime_error("KimiK3ShortConv: channels must be divisible by tp_size");
    }
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
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
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
    if (!model_config) {
        throw std::invalid_argument("KimiK3DeltaAttention: model config is null");
    }
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const auto &linear_config = model_config->get_config_json().at("linear_attn_config");
    const size_t num_heads = linear_config.at("num_heads").get<size_t>();
    head_dim_ = linear_config.at("head_dim").get<size_t>();
    conv_kernel_size_ = linear_config.at("short_conv_kernel_size").get<size_t>();
    gate_lower_bound_ = linear_config.value("gate_lower_bound", -5.0f);
    const double eps = model_config->get<double>("rms_norm_eps");
    const auto tp_size = static_cast<size_t>(rank_info.tp_size);
    if (!hidden_size || !num_heads || !head_dim_ || !conv_kernel_size_
        || !std::isfinite(eps) || eps <= 0) {
        throw std::invalid_argument("KimiK3DeltaAttention: invalid construction geometry/config");
    }
    if (num_heads % tp_size != 0) {
        throw std::runtime_error("KimiK3DeltaAttention: num_heads must be divisible by tp_size");
    }
    if (rank_info.tp_size > 1 && !rank_info.comm) {
        throw std::invalid_argument("KimiK3DeltaAttention: TP communicator is missing");
    }
    const auto max = std::numeric_limits<size_t>::max();
    if (num_heads > max / head_dim_ || num_heads * head_dim_ > max / 3) {
        throw std::length_error("KimiK3DeltaAttention: projection size overflows");
    }
    local_num_heads_ = num_heads / tp_size;
    local_projection_size_ = local_num_heads_ * head_dim_;
    const size_t projection_size = num_heads * head_dim_;
    // QKV, ShortConv, replicated/column/row Linear and scalar/norm layouts.
    // Every eventual allocation is preflighted before the first QKV child.
    checked_parameter_geometry({3 * local_projection_size_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, 1, conv_kernel_size_}, dtype);
    checked_parameter_geometry({3 * local_projection_size_, 1, conv_kernel_size_}, dtype);
    checked_parameter_geometry({local_num_heads_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({head_dim_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, head_dim_}, dtype);
    checked_parameter_geometry({local_projection_size_}, infinilm::runtime::DataType::kFloat32);
    checked_parameter_geometry({local_num_heads_, hidden_size}, dtype);
    checked_parameter_geometry({local_projection_size_, hidden_size}, dtype);
    checked_parameter_geometry({head_dim_}, dtype);
    checked_parameter_geometry({hidden_size, local_projection_size_}, dtype);
    device_ = runtime.device();
    conv_dtype_ = dtype;

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
            runtime, projection_size, conv_kernel_size_, dtype, rank_info);
        this->bind_child("q_conv1d", q_conv1d_);
        k_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, dtype, rank_info);
        this->bind_child("k_conv1d", k_conv1d_);
        v_conv1d_ = std::make_shared<KimiK3ShortConv>(
            runtime, projection_size, conv_kernel_size_, dtype, rank_info);
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
        o_norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, head_dim_, eps, dtype);
        this->bind_child("o_norm", o_norm_);
        o_proj_ = std::make_shared<infinilm::layers::linear::RowParallelLinear>(
            runtime, projection_size, hidden_size, false, dtype,
            rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);
    } catch (...) {
        const auto failure = std::current_exception();
        // All constructed members/registered aliases are still alive here.
        // Leaf constructor failures keep their existing same-owner contract.
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

void KimiK3DeltaAttention::process_weights_after_loading(
    infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_) {
        throw std::invalid_argument("KimiK3DeltaAttention: postload Runtime device differs");
    }
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("KimiK3DeltaAttention: postload requires inactive graph state");
    }
    if (postload_attempted_) {
        throw std::logic_error("KimiK3DeltaAttention: postload is single-use after mutation attempt");
    }
    if (!local_projection_size_ || !conv_kernel_size_
        || local_projection_size_ > std::numeric_limits<size_t>::max() / 3) {
        throw std::invalid_argument("KimiK3DeltaAttention: invalid convolution pack geometry");
    }

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
        if (weight->device() != runtime.device() || weight->dtype() != conv_dtype_) {
            throw std::invalid_argument("KimiK3DeltaAttention: convolution weight device/dtype differs");
        }
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
        // Projection finalization is owned by loading_children traversal.
        pending = infinilm::runtime::Tensor::empty(
            runtime, {3 * local_projection_size_, 1, conv_kernel_size_}, conv_dtype_);
        for (size_t i = 0; i < slices.size(); ++i) {
            slices[i] = pending->narrow({{0, i * local_projection_size_, local_projection_size_}});
        }
        for (size_t i = 0; i < weights.size(); ++i) {
            slices[i]->copy_from(runtime, weights[i]);
        }
        // Finite same-owner stream fence, once at this postload boundary.
        // Sources, old/new QKV bindings, old pack, pack and all slices live here.
        runtime.syncStream();
        packed_conv_weight_ = std::move(pending);
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        // Preserve the first error. Hidden aliases may already be transformed;
        // the worker must remain Failed and this owner cannot be retried.
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor KimiK3DeltaAttention::forward(
    const infinilm::runtime::Tensor &hidden_states,
    global_state::ForwardContext &forward_context) const {
    throw std::runtime_error("Kimi delta attention requires an InfiniOps KDA implementation");
}

} // namespace infinilm::models::kimi_k3
