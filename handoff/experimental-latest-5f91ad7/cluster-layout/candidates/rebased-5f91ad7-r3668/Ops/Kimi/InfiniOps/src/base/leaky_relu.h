#ifndef INFINI_OPS_BASE_LEAKY_RELU_H_
#define INFINI_OPS_BASE_LEAKY_RELU_H_

#include "operator.h"

namespace infini::ops {

class LeakyRelu : public Operator<LeakyRelu> {
 public:
  LeakyRelu(const Tensor input, const double negative_slope, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        negative_slope_{negative_slope},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const double negative_slope,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double negative_slope_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
