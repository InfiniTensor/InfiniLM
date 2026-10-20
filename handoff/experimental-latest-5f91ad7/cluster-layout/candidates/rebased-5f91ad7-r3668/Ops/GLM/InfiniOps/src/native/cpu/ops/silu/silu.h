#ifndef INFINI_OPS_CPU_SILU_SILU_H_
#define INFINI_OPS_CPU_SILU_SILU_H_

#include <cmath>

#include "base/silu.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kCpu> : public Silu,
                                           Caster<Device::Type::kCpu> {
 public:
  using Silu::Silu;

  void operator()(const Tensor input, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, out);
        },
        "Operator<Silu, Device::Type::kCpu>::operator()");
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
      const auto input_idx = get_idx(
          i, is_input_contiguous_, input_shape_.data(), input_strides_.data());
      const auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                                   out_strides_.data());
      const ComputeType input_value = Cast<ComputeType>(input_ptr[input_idx]);
      const ComputeType sigmoid = static_cast<ComputeType>(
          1.0 / (1.0 + std::exp(-static_cast<double>(input_value))));
      out_ptr[out_idx] = Cast<T>(input_value * sigmoid);
    }
  }
};

}  // namespace infini::ops

#endif
