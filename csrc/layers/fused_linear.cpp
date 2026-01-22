#include "fused_linear.hpp"

#include <spdlog/spdlog.h>

namespace infinilm::layers {
// ---------------------------------------------------------
// QKV Parallel Linear
// ---------------------------------------------------------
QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head,
                                     size_t num_kv_head,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info,
                                     std::optional<infinicore::nn::QuantScheme> quant_scheme)
    : QKVParallelLinear(hidden_size,
                        head_dim, head_dim, head_dim,
                        num_q_head, num_kv_head, num_kv_head,
                        bias, bias, bias,
                        dtype, device, rank_info,
                        quant_scheme) {}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info,
                                     std::optional<infinicore::nn::QuantScheme> quant_scheme)
    : infinicore::nn::ColumnParallelLinear(
          hidden_size,
          num_q_head * q_dim + num_k_head * k_dim + num_v_head * v_dim,
          (q_bias || k_bias || v_bias),
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size,
          quant_scheme),
      q_dim_(q_dim),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_q_head_(num_q_head),
      num_k_head_(num_k_head),
      num_v_head_(num_v_head),
      q_bias_(q_bias),
      k_bias_(k_bias),
      v_bias_(v_bias) {
    if (num_q_head % tp_size_ != 0 || num_k_head % tp_size_ != 0 || num_v_head % tp_size_ != 0) {
        throw std::runtime_error("QKVParallelLinear: num_[q|k|v]_head must be divisible by tp_size");
    }

    if ((q_bias_ != k_bias_) || (k_bias_ != v_bias_)) {
        throw std::runtime_error("q_bias, k_bias, v_bias must all match");
    }

    q_out_size_ = num_q_head_ * q_dim_ / tp_size_;
    k_out_size_ = num_k_head_ * k_dim_ / tp_size_;
    v_out_size_ = num_v_head_ * v_dim_ / tp_size_;
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
QKVParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);

    auto q_out = output->narrow({{2, 0, q_out_size_}});
    auto k_out = output->narrow({{2, q_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, k_out, v_out);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight() const {
    return infinicore::nn::Parameter(
        weight_->narrow({{0, 0, q_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight() const {
    return infinicore::nn::Parameter(
        weight_->narrow({{0, q_out_size_, k_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight() const {
    return infinicore::nn::Parameter(
        weight_->narrow({{0, q_out_size_ + k_out_size_, v_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight_scale() const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{0, 0, q_out_size_}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight_scale() const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{0, q_out_size_, k_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight_scale() const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{0, q_out_size_ + k_out_size_, v_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_bias() const {
    if (!q_bias_) {
        return infinicore::nn::Parameter();
    }
    return infinicore::nn::Parameter(
        bias_->narrow({{0, 0, q_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_bias() const {
    if (!k_bias_) {
        return infinicore::nn::Parameter();
    }
    return infinicore::nn::Parameter(
        bias_->narrow({{0, q_out_size_, k_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_bias() const {
    if (!v_bias_) {
        return infinicore::nn::Parameter();
    }
    return infinicore::nn::Parameter(
        bias_->narrow({{0, q_out_size_ + k_out_size_, v_out_size_}}),
        0, tp_rank_, tp_size_);
}

bool QKVParallelLinear::has_q_bias() const { return q_bias_; }
bool QKVParallelLinear::has_k_bias() const { return k_bias_; }
bool QKVParallelLinear::has_v_bias() const { return v_bias_; }

// ---------------------------------------------------------
// Gate-Up Parallel Linear
// ---------------------------------------------------------
GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info,
                                           std::optional<infinicore::nn::QuantScheme> quant_scheme)
    : GateUpParallelLinear(hidden_size, intermediate_size, bias, bias, dtype, device, rank_info, quant_scheme) {
}

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info,
                                           std::optional<infinicore::nn::QuantScheme> quant_scheme)
    : infinicore::nn::ColumnParallelLinear(hidden_size, intermediate_size * 2, gate_bias || up_bias, dtype, device, rank_info.tp_rank, rank_info.tp_size, quant_scheme), gate_bias_(gate_bias), up_bias_(up_bias) {
    if (gate_bias_ != up_bias_) {
        throw std::runtime_error("Not supported yet: gate_bias and up_bias should be given at the same time");
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> GateUpParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);
    auto cols = output->shape()[2];
    auto gate_output = output->narrow({{2, 0, cols / 2}});
    auto up_output = output->narrow({{2, cols / 2, cols / 2}});
    return std::make_tuple(gate_output, up_output);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight() const {
    return infinicore::nn::Parameter(weight_->narrow({{0, 0, weight_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_bias() const {
    if (!gate_bias_) {
        return infinicore::nn::Parameter();
    } else {
        return infinicore::nn::Parameter(bias_->narrow({{0, 0, bias_->size(0) / 2}}), 0, tp_rank_, tp_size_);
    }
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight() const {
    return infinicore::nn::Parameter(weight_->narrow({{0, weight_->size(0) / 2, weight_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_bias() const {
    if (!up_bias_) {
        return infinicore::nn::Parameter();
    } else {
        return infinicore::nn::Parameter(bias_->narrow({{0, bias_->size(0) / 2, bias_->size(0) / 2}}),
                                         0, tp_rank_, tp_size_);
    }
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight_scale() const {
    return infinicore::nn::Parameter(weight_scale_->narrow({{0, 0, weight_scale_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight_scale() const {
    return infinicore::nn::Parameter(weight_scale_->narrow({{0, weight_scale_->size(0) / 2, weight_scale_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

bool GateUpParallelLinear::has_gate_bias() const {
    return gate_bias_;
}

bool GateUpParallelLinear::has_up_bias() const {
    return up_bias_;
}
} // namespace infinilm::layers
