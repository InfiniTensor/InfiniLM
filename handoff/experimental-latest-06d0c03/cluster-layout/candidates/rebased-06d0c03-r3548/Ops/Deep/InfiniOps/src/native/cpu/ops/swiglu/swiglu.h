#ifndef INFINI_OPS_CPU_SWIGLU_SWIGLU_H_
#define INFINI_OPS_CPU_SWIGLU_SWIGLU_H_

#include <cmath>

#include "base/swiglu.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kCpu> : public Swiglu,
                                             Caster<Device::Type::kCpu> {
 public:
  using Swiglu::Swiglu;

  void operator()(const Tensor input, const Tensor gate,
                  Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, gate, out);
        },
        "Operator<Swiglu, Device::Type::kCpu>::operator()");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, const Tensor gate, Tensor out) const {
    using ComputeType = std::conditional_t<IsBFloat16<Device::Type::kCpu, T> ||
                                               IsFP16<Device::Type::kCpu, T>,
                                           float, T>;

    const auto* input_ptr = static_cast<const T*>(input.data());
    const auto* gate_ptr = static_cast<const T*>(gate.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto input_idx = get_idx(i, is_input_contiguous_, input_shape_.data(),
                               input_strides_.data());
      auto gate_idx = get_idx(i, is_gate_contiguous_, gate_shape_.data(),
                              gate_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());
      const ComputeType gate_val = Cast<ComputeType>(gate_ptr[gate_idx]);
      const ComputeType sigmoid_gate = static_cast<ComputeType>(
          1.0 / (1.0 + std::exp(-static_cast<double>(gate_val))));
      const ComputeType swish_gate = gate_val * sigmoid_gate;
      out_ptr[out_idx] =
          Cast<T>(Cast<ComputeType>(input_ptr[input_idx]) * swish_gate);
    }
  }
};

}  // namespace infini::ops

#endif
