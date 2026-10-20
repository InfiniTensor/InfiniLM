#ifndef INFINI_OPS_BASE_QUICK_GELU_H_
#define INFINI_OPS_BASE_QUICK_GELU_H_

#include "operator.h"

namespace infini::ops {

// Inference y = x * sigmoid(1.702f * x). Native providers specify the
// dtype-specific intermediate rounding; this is not Gelu's none/tanh mode.
class QuickGelu : public Operator<QuickGelu> {
 public:
  static constexpr bool kAutomaticDefault = true;

  QuickGelu(const Tensor input, Tensor out)
      : input_shape_{input.shape()}, input_strides_{input.strides()},
        out_shape_{out.shape()}, out_strides_{out.strides()},
        dtype_{input.dtype()}, device_{input.device()} {}

  virtual void operator()(const Tensor input, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_, out_shape_;
  Tensor::Strides input_strides_, out_strides_;
  DataType dtype_;
  Device device_;
};

}  // namespace infini::ops
#endif
