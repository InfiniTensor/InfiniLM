#ifndef INFINI_OPS_CPU_MUL_MUL_H_
#define INFINI_OPS_CPU_MUL_MUL_H_

#include <utility>

#include "base/mul.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kCpu> : public Mul,
                                          Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Mul{input, other, out} {}

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, other, out);
        },
        "`Operator<Mul, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, const Tensor other, Tensor out) const {
    using ComputeType = std::conditional_t<IsBFloat16<Device::Type::kCpu, T> ||
                                               IsFP16<Device::Type::kCpu, T>,
                                           float, T>;

    const auto* input_ptr = static_cast<const T*>(input.data());
    const auto* other_ptr = static_cast<const T*>(other.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto input_idx = get_idx(i, is_input_contiguous_, input_shape_.data(),
                               input_strides_.data());
      auto other_idx = get_idx(i, is_other_contiguous_, other_shape_.data(),
                               other_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());

      out_ptr[out_idx] = Cast<T>(Cast<ComputeType>(input_ptr[input_idx]) *
                                 Cast<ComputeType>(other_ptr[other_idx]));
    }
  }
};

}  // namespace infini::ops

#endif
