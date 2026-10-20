#ifndef INFINI_OPS_CPU_RELU_RELU_H_
#define INFINI_OPS_CPU_RELU_RELU_H_

#include <type_traits>
#include <vector>

#include "base/relu.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kCpu> : public Relu,
                                           Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor input, Tensor out) : Relu{input, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, ReluDataTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, out);
        },
        "`Operator<Relu, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, Tensor out) const {
    using ComputeType = std::conditional_t<IsBFloat16<Device::Type::kCpu, T> ||
                                               IsFP16<Device::Type::kCpu, T>,
                                           float, T>;

    const auto* input_ptr = static_cast<const T*>(input.data());
    auto* out_ptr = static_cast<T*>(out.data());

    auto get_idx = [&](Tensor::Size i, bool is_contiguous, const auto* shape,
                       const auto* strides) {
      return is_contiguous ? i : utils::IndexToOffset(i, ndim_, shape, strides);
    };

    std::vector<T> input_copy;
    bool input_contiguous = is_input_contiguous_;

    if (NeedsInputCopy(input, out)) {
      input_copy.resize(output_size_);

#pragma omp parallel for
      for (Tensor::Size i = 0; i < output_size_; ++i) {
        auto input_idx = get_idx(i, is_input_contiguous_, input_shape_.data(),
                                 input_strides_.data());
        input_copy[i] = input_ptr[input_idx];
      }

      input_ptr = input_copy.data();
      input_contiguous = true;
    }

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto input_idx = get_idx(i, input_contiguous, input_shape_.data(),
                               input_strides_.data());
      auto out_idx = get_idx(i, is_out_contiguous_, out_shape_.data(),
                             out_strides_.data());
      auto value = Cast<ComputeType>(input_ptr[input_idx]);

      if constexpr (std::is_unsigned_v<ComputeType>) {
        out_ptr[out_idx] = input_ptr[input_idx];
      } else {
        out_ptr[out_idx] = value < ComputeType{0} ? Cast<T>(ComputeType{0})
                                                  : input_ptr[input_idx];
      }
    }
  }
};

}  // namespace infini::ops

#endif
