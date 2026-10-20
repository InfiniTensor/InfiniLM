#ifndef INFINI_OPS_BASE_HARDSHRINK_BACKWARD_H_
#define INFINI_OPS_BASE_HARDSHRINK_BACKWARD_H_

#include "operator.h"

namespace infini::ops {

class HardshrinkBackward : public Operator<HardshrinkBackward> {
 public:
  HardshrinkBackward(const Tensor grad_out, const Tensor input,
                     const double lambd, Tensor grad_input)
      : grad_out_shape_{grad_out.shape()},
        grad_out_strides_{grad_out.strides()},
        grad_out_type_{grad_out.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        lambd_{lambd},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_out, const Tensor input,
                          const double lambd, Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_out_shape_;

  Tensor::Strides grad_out_strides_;

  DataType grad_out_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  double lambd_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
