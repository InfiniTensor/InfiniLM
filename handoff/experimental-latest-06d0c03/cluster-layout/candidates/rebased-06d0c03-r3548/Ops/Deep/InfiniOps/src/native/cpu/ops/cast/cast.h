#ifndef INFINI_OPS_CPU_CAST_CAST_H_
#define INFINI_OPS_CPU_CAST_CAST_H_

#include "base/cast.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Cast, Device::Type::kCpu> : public Cast {
 public:
  Operator(const Tensor input, Tensor out) : Cast{input, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllTypes, AllTypes>(
        {input_dtype_, out_dtype_},
        [&](auto in_tag, auto out_tag) {
          using InT = typename decltype(in_tag)::type;
          using OutT = typename decltype(out_tag)::type;
          Compute<InT, OutT>(input, out);
        },
        "`Operator<Cast, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename InT, typename OutT>
  void Compute(const Tensor input, Tensor out) const {
    const auto* in_ptr = static_cast<const InT*>(input.data());
    auto* out_ptr = static_cast<OutT*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto in_idx = get_idx(i, is_input_contiguous_, input_shape_.data(),
                            input_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());

      out_ptr[out_idx] =
          Caster<Device::Type::kCpu>::template Cast<OutT>(in_ptr[in_idx]);
    }
  }
};

}  // namespace infini::ops

#endif
