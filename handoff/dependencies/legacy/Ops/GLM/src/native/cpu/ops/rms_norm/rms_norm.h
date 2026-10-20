#ifndef INFINI_OPS_CPU_RMS_NORM_H_
#define INFINI_OPS_CPU_RMS_NORM_H_

#include <cmath>

#include "base/rms_norm.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "native/cpu/caster_.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kCpu> : public RmsNorm,
                                              Caster<Device::Type::kCpu> {
 public:
  using RmsNorm::RmsNorm;

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, weight, eps, out);
        },
        "`Operator<RmsNorm, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, const Tensor weight, float eps,
               Tensor out) const {
    auto* out_ptr = static_cast<T*>(out.data());
    const auto* input_ptr = static_cast<const T*>(input.data());
    const auto* weight_ptr = static_cast<const T*>(weight.data());

    auto stride_input_batch = input_strides_.size() > 1 ? input_strides_[0] : 0;
    auto stride_input_nhead =
        input_strides_.size() > 1 ? input_strides_[1] : input_strides_[0];
    auto stride_out_batch = out_strides_.size() > 1 ? out_strides_[0] : 0;
    auto stride_out_nhead =
        out_strides_.size() > 1 ? out_strides_[1] : out_strides_[0];

    for (Tensor::Size bi = 0; bi < batch_size_; ++bi) {
      for (Tensor::Size hi = 0; hi < nhead_; ++hi) {
        const T* input_row =
            input_ptr + bi * stride_input_batch + hi * stride_input_nhead;
        T* out_row = out_ptr + bi * stride_out_batch + hi * stride_out_nhead;

        float ss = 0;
        for (Tensor::Size k = 0; k < dim_; ++k) {
          float v = Cast<float>(input_row[k]);
          ss += v * v;
        }
        float rms = 1.f / std::sqrt(ss / static_cast<float>(dim_) + eps);

        for (Tensor::Size k = 0; k < dim_; ++k) {
          out_row[k] = Cast<T>(Cast<float>(input_row[k]) *
                               Cast<float>(weight_ptr[k]) * rms);
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
