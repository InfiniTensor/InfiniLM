#ifndef INFINI_OPS_CPU_CAUSAL_SOFTMAX_H_
#define INFINI_OPS_CPU_CAUSAL_SOFTMAX_H_

#include <cmath>

#include "base/causal_softmax.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "native/cpu/caster_.h"
#include "tensor.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kCpu> : public CausalSoftmax,
                                                    Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor input, Tensor out) : CausalSoftmax{input, out} {}

  void operator()(const Tensor input, Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, out);
        },
        "`Operator<CausalSoftmax, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, Tensor out) const {
    auto* out_ptr = static_cast<T*>(out.data());
    const auto* input_ptr = static_cast<const T*>(input.data());

    auto out_stride_b = ndim_ == 3 ? out_strides_[0] : 0;
    auto out_stride_i = out_strides_[ndim_ - 2];
    auto out_stride_j = out_strides_[ndim_ - 1];
    auto input_stride_b = ndim_ == 3 ? input_strides_[0] : 0;
    auto input_stride_i = input_strides_[ndim_ - 2];
    auto input_stride_j = input_strides_[ndim_ - 1];

    for (Tensor::Size bi = 0; bi < batch_size_; ++bi) {
      for (Tensor::Size i = 0; i < seq_len_; ++i) {
        ptrdiff_t out_offset = bi * out_stride_b + i * out_stride_i;
        ptrdiff_t input_offset = bi * input_stride_b + i * input_stride_i;
        T* out_row = out_ptr + out_offset;
        const T* input_row = input_ptr + input_offset;

        Tensor::Size valid_len = total_seq_len_ - seq_len_ + i + 1;

        for (Tensor::Size j = valid_len; j < total_seq_len_; ++j) {
          out_row[j * out_stride_j] = Cast<T>(0.0f);
        }

        float max_val = Cast<float>(input_row[0]);
        for (Tensor::Size j = 1; j < valid_len; ++j) {
          float v = Cast<float>(input_row[j * input_stride_j]);
          if (v > max_val) {
            max_val = v;
          }
        }

        float sum = 0.0f;
        for (Tensor::Size j = 0; j < valid_len; ++j) {
          float v =
              std::exp(Cast<float>(input_row[j * input_stride_j]) - max_val);
          out_row[j * out_stride_j] = Cast<T>(v);
          sum += v;
        }

        for (Tensor::Size j = 0; j < valid_len; ++j) {
          out_row[j * out_stride_j] =
              Cast<T>(Cast<float>(out_row[j * out_stride_j]) / sum);
        }
      }
    }
  }
};

}  // namespace infini::ops

#endif
