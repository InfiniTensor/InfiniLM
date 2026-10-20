#ifndef INFINI_OPS_BASE_MULTINOMIAL_H_
#define INFINI_OPS_BASE_MULTINOMIAL_H_

#include "operator.h"

namespace infini::ops {

class Multinomial : public Operator<Multinomial> {
 public:
  Multinomial(const Tensor input, const int64_t num_samples,
              const bool replacement, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        num_samples_{num_samples},
        replacement_{replacement},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t num_samples,
                          const bool replacement, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t num_samples_{};

  bool replacement_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
