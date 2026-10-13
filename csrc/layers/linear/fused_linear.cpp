#include "fused_linear.hpp"

#include <spdlog/spdlog.h>
#include "runtime/runtime.hpp"
#include <exception>
#include <limits>
#include <stdexcept>

namespace infinilm::layers::linear {

namespace {
void validate_fused_rank(infinilm::runtime::Runtime &runtime, const engine::distributed::RankInfo &rank) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Fused Linear construction requires eager Runtime");
    if (rank.device != runtime.device() || rank.tp_size <= 0 || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size)
        throw std::invalid_argument("Invalid explicit fused Linear rank/device");
}
size_t checked_qkv_output(infinilm::runtime::Runtime &runtime, size_t hidden,
                          size_t nq, size_t dq, size_t nk, size_t dk, size_t nv, size_t dv,
                          bool qb, bool kb, bool vb, const engine::distributed::RankInfo &rank) {
    validate_fused_rank(runtime, rank);
    const size_t tp = static_cast<size_t>(rank.tp_size);
    if (!hidden || !nq || !nk || !nv || !dq || !dk || !dv || nq % tp
        || (nk % tp && tp % nk) || (nv % tp && tp % nv) || qb != kb || kb != vb)
        throw std::invalid_argument("Invalid explicit QKV geometry/bias");
    const size_t rk = nk % tp == 0 ? 1 : (tp + nk - 1) / nk;
    const size_t rv = nv % tp == 0 ? 1 : (tp + nv - 1) / nv;
    const size_t max = std::numeric_limits<size_t>::max();
    if (nq > max / dq || nk > max / dk || nv > max / dv
        || nk * dk > max / rk || nv * dv > max / rv)
        throw std::length_error("Explicit QKV output size overflow");
    const size_t q = nq * dq, k = nk * dk * rk, v = nv * dv * rv;
    if (q > max - k || q + k > max - v) throw std::length_error("Explicit QKV output size overflow");
    return q + k + v;
}
size_t checked_gate_output(infinilm::runtime::Runtime &runtime, size_t hidden, size_t intermediate,
                           bool gate_bias, bool up_bias, const engine::distributed::RankInfo &rank) {
    validate_fused_rank(runtime, rank);
    if (!hidden || !intermediate || intermediate % static_cast<size_t>(rank.tp_size)
        || gate_bias != up_bias) throw std::invalid_argument("Invalid explicit GateUp geometry/bias");
    if (intermediate > std::numeric_limits<size_t>::max() / 2)
        throw std::length_error("Explicit GateUp output size overflow");
    return intermediate * 2;
}
} // namespace
QKVParallelLinear::QKVParallelLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head,
                                     size_t num_kv_head,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinilm::runtime::DataType &dtype,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(runtime, hidden_size,
                        head_dim, head_dim, head_dim,
                        num_q_head, num_kv_head, num_kv_head,
                        bias, bias, bias,
                        quantization,
                        dtype, rank_info) {}

QKVParallelLinear::QKVParallelLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     const infinilm::runtime::DataType &dtype,
                                     engine::distributed::RankInfo rank_info)
    : infinilm::nn::ColumnParallelLinear(
        runtime, hidden_size,
        checked_qkv_output(runtime, hidden_size, num_q_head, q_dim, num_k_head, k_dim, num_v_head, v_dim, q_bias, k_bias, v_bias, rank_info),
        quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
        (q_bias || k_bias || v_bias),
        dtype,
        rank_info.tp_rank,
        rank_info.tp_size),
      q_dim_(q_dim),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_q_head_(num_q_head),
      num_k_head_(num_k_head),
      num_v_head_(num_v_head),
      q_bias_(q_bias),
      k_bias_(k_bias),
      v_bias_(v_bias),
      num_kv_head_replicas_(calculate_kv_replicas(num_k_head, rank_info.tp_size)) {
    q_out_size_ = num_q_head_ * q_dim_ / tp_size_;
    k_out_size_ = num_kv_head_replicas_ * num_k_head_ * k_dim_ / tp_size_;
    v_out_size_ = num_kv_head_replicas_ * num_v_head_ * v_dim_ / tp_size_;
}

QKVParallelLinear::QKVParallelLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head, size_t num_kv_head,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinilm::runtime::DataType &dtype,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(runtime, hidden_size, head_dim, head_dim, head_dim, num_q_head, num_kv_head, num_kv_head, bias, bias, bias, q_name, k_name, v_name, register_fn, quantization, dtype, rank_info) {
}

QKVParallelLinear::QKVParallelLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     const std::string &q_name, const std::string &k_name, const std::string &v_name,
                                     RegisterParamFn register_fn,
                                     std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                     const infinilm::runtime::DataType &dtype,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(runtime, hidden_size, q_dim, k_dim, v_dim, num_q_head, num_k_head, num_v_head, q_bias, k_bias, v_bias, quantization, dtype, rank_info) {
    try {
    register_fn_ = register_fn;
    split_infos_ = {
        {q_name, 0, q_out_size_, 0},
        {k_name, q_out_size_, k_out_size_, num_k_head_},
        {v_name, q_out_size_ + k_out_size_, v_out_size_, num_v_head_},
    };
    auto params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }

    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

GateUpParallelLinear::GateUpParallelLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size, size_t intermediate_size, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
                                           const infinilm::runtime::DataType &dtype,
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(runtime, hidden_size, intermediate_size, bias, bias, quantization, dtype, rank_info) {
}

GateUpParallelLinear::GateUpParallelLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           const infinilm::runtime::DataType &dtype,
                                           engine::distributed::RankInfo rank_info)
    : infinilm::nn::ColumnParallelLinear(
        runtime, hidden_size,
        checked_gate_output(runtime, hidden_size, intermediate_size, gate_bias, up_bias, rank_info),
        quantization == nullptr ? std::make_shared<infinilm::quantization::NoneQuantization>() : quantization,
        gate_bias || up_bias,
        dtype,
        rank_info.tp_rank,
        rank_info.tp_size),
      gate_bias_(gate_bias),
      up_bias_(up_bias) {}

GateUpParallelLinear::GateUpParallelLinear(infinilm::runtime::Runtime &runtime, size_t hidden_size, size_t intermediate_size,
                                           const std::string &gate_name, const std::string &up_name,
                                           RegisterParamFn register_fn,
                                           std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                                           bool bias,
                                           const infinilm::runtime::DataType &dtype,
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(runtime, hidden_size, intermediate_size, quantization, bias, dtype, rank_info) {
    try {
    const std::string &key_name = local_weights_.count("qweight") ? "qweight" : "weight";
    const auto &key_param = get_parameter_ref(key_name);
    int fused_dim = this->get_quantization()->get_fused_split_dim();
    size_t logical_output = this->get_quantization()->get_logical_dim_size(key_param->size(fused_dim));
    size_t half_size = logical_output / 2;
    register_fn_ = register_fn;
    split_infos_ = {
        {gate_name, 0, half_size},
        {up_name, half_size, half_size},
    };
    auto params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
    for (auto &sp : params) {
        register_fn_(sp.full_name, std::move(sp.param));
    }

    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}
// ---------------------------------------------------------
// QKV Parallel Linear
// ---------------------------------------------------------

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor, infinilm::runtime::Tensor>
QKVParallelLinear::forward_split(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) {
    auto output = this->forward(runtime, input);

    auto q_out = output->narrow({{2, 0, q_out_size_}});
    auto k_out = output->narrow({{2, q_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, k_out, v_out);
}

bool QKVParallelLinear::has_q_bias() const { return q_bias_; }
bool QKVParallelLinear::has_k_bias() const { return k_bias_; }
bool QKVParallelLinear::has_v_bias() const { return v_bias_; }

void QKVParallelLinear::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("postload Runtime and Linear devices differ");
    if (pre_transpose && runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("dense pre_transpose requires inactive graph state");
    // Old aliases and new split views must both outlive catch-side owner drain.
    auto old_parameters = local_weights_;
    auto old_bound = bound_params_;
    auto old_quant = quantization_;
    std::vector<infinilm::quantization::SplitParam> params;
    try {
        BaseLinear::process_weights_after_loading(runtime, pre_transpose);
    if (register_fn_ && !split_infos_.empty()) {
            params = this->split_params(split_infos_, tp_rank_, tp_size_, num_k_head_);
        for (auto &sp : params) {
            register_fn_(sp.full_name, std::move(sp.param));
        }
    }
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

// ---------------------------------------------------------
// Gate-Up Parallel Linear
// ---------------------------------------------------------

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> GateUpParallelLinear::forward_split(
    infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) {
    auto output = this->forward(runtime, input);
    auto cols = output->shape()[2];
    auto gate_output = output->narrow({{2, 0, cols / 2}});
    auto up_output = output->narrow({{2, cols / 2, cols / 2}});
    return std::make_tuple(gate_output, up_output);
}

bool GateUpParallelLinear::has_gate_bias() const { return gate_bias_; }
bool GateUpParallelLinear::has_up_bias() const { return up_bias_; }

void GateUpParallelLinear::process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) {
    runtime.assertOwnerThread();
    if (runtime.device() != device_)
        throw std::invalid_argument("postload Runtime and Linear devices differ");
    if (pre_transpose && runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive)
        throw std::logic_error("dense pre_transpose requires inactive graph state");
    // Old aliases and new split views must both outlive catch-side owner drain.
    auto old_parameters = local_weights_;
    auto old_bound = bound_params_;
    auto old_quant = quantization_;
    std::vector<infinilm::quantization::SplitParam> params;
    try {
        BaseLinear::process_weights_after_loading(runtime, pre_transpose);
        if (register_fn_ && !split_infos_.empty()) {
            params = this->split_params(split_infos_, tp_rank_, tp_size_, -1);
            for (auto &sp : params) {
                register_fn_(sp.full_name, std::move(sp.param));
            }
        }
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

} // namespace infinilm::layers::linear
