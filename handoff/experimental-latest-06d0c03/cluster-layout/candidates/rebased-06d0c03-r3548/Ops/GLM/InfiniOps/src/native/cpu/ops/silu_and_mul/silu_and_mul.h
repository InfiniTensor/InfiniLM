#ifndef INFINI_OPS_CPU_SILU_AND_MUL_SILU_AND_MUL_H_
#define INFINI_OPS_CPU_SILU_AND_MUL_SILU_AND_MUL_H_

#include <cmath>

#include "base/silu_and_mul.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kCpu> : public SiluAndMul,
                                                 Caster<Device::Type::kCpu> {
 public:
  using SiluAndMul::SiluAndMul;

  void operator()(const Tensor input, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, out);
        },
        "Operator<SiluAndMul, Device::Type::kCpu>::operator()");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, Tensor out) const {
    using ComputeType = std::conditional_t<IsBFloat16<Device::Type::kCpu, T> ||
                                               IsFP16<Device::Type::kCpu, T>,
                                           float, T>;

    const auto* input_ptr = static_cast<const T*>(input.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      const auto row = i / hidden_size_;
      const auto column = i % hidden_size_;
      const auto gate_logical_idx = row * 2 * hidden_size_ + column;
      const auto up_logical_idx = gate_logical_idx + hidden_size_;
      auto gate_idx = get_idx(gate_logical_idx, is_input_contiguous_,
                              input_shape_.data(), input_strides_.data());
      auto up_idx = get_idx(up_logical_idx, is_input_contiguous_,
                            input_shape_.data(), input_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());
      const ComputeType gate_val = Cast<ComputeType>(input_ptr[gate_idx]);
      const ComputeType sigmoid_gate = static_cast<ComputeType>(
          1.0 / (1.0 + std::exp(-static_cast<double>(gate_val))));
      const ComputeType swish_gate = gate_val * sigmoid_gate;
      out_ptr[out_idx] =
          Cast<T>(swish_gate * Cast<ComputeType>(input_ptr[up_idx]));
    }
  }
};

}  // namespace infini::ops

#endif
