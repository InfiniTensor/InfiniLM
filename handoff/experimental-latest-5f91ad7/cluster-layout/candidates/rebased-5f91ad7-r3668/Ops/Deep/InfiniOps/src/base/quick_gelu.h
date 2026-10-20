#ifndef INFINI_OPS_BASE_QUICK_GELU_H_
#define INFINI_OPS_BASE_QUICK_GELU_H_

#include <functional>

#include "operator.h"

namespace infini::ops {

// Inference y = x * sigmoid(1.702f * x). Native providers specify the
// dtype-specific intermediate rounding; this is not Gelu's none/tanh mode.
class QuickGelu : public Operator<QuickGelu> {
 public:
  static constexpr bool kAutomaticDefault = true;

  QuickGelu(const Tensor input, Tensor out)
      : input_meta_{input}, out_meta_{out}, device_{input.device()} {}

  virtual void operator()(const Tensor input, Tensor out) const = 0;

 protected:
  const Tensor input_meta_, out_meta_;
  Device device_;
};

}  // namespace infini::ops
#endif
