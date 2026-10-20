#ifndef INFINI_OPS_CPU_LINEAR_LINEAR_H_
#define INFINI_OPS_CPU_LINEAR_LINEAR_H_

#include <utility>

#include "base/linear.h"
#include "common/generic_utils.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<Linear, Device::Type::kCpu> : public Linear,
                                             Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           Tensor out)
      : Linear{input, weight, bias, out} {}

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, weight, bias, out);
        },
        "`Operator<Linear, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, const Tensor weight,
               std::optional<Tensor> bias, Tensor out) const {
    const auto* input_ptr = static_cast<const T*>(input.data());
    const auto* weight_ptr = static_cast<const T*>(weight.data());
    auto* out_ptr = static_cast<T*>(out.data());
    const T* bias_ptr = bias ? static_cast<const T*>(bias->data()) : nullptr;

    auto in_features = input.size(-1);
    auto out_features = weight.size(0);

    for (Tensor::Size row = 0; row < rows_; ++row) {
      auto remaining = row;
      Tensor::Stride input_base = 0;
      Tensor::Stride output_base = 0;

      for (Tensor::Size axis = input.ndim() - 1; axis > 0; --axis) {
        auto leading_axis = axis - 1;
        auto coordinate = remaining % input.size(leading_axis);
        remaining /= input.size(leading_axis);
        input_base += coordinate * input.stride(leading_axis);
        output_base += coordinate * out.stride(leading_axis);
      }

      for (Tensor::Size output_feature = 0; output_feature < out_features;
           ++output_feature) {
        float sum = 0.0f;

        for (Tensor::Size input_feature = 0; input_feature < in_features;
             ++input_feature) {
          auto input_value =
              input_ptr[input_base + input_feature * input.stride(-1)];
          auto weight_value = weight_ptr[output_feature * weight.stride(0) +
                                         input_feature * weight.stride(1)];
          sum += Cast<float>(input_value) * Cast<float>(weight_value);
        }

        if (bias_ptr) {
          sum += Cast<float>(bias_ptr[output_feature * bias->stride(0)]);
        }

        out_ptr[output_base + output_feature * out.stride(-1)] = Cast<T>(sum);
      }
    }
  }
};

}  // namespace infini::ops

#endif
