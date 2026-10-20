#ifndef INFINI_OPS_BASE_SIGMOID_BACKWARD_H_
#define INFINI_OPS_BASE_SIGMOID_BACKWARD_H_

#include "operator.h"

namespace infini::ops {

class SigmoidBackward : public Operator<SigmoidBackward> {
 public:
  SigmoidBackward(const Tensor grad_output, const Tensor output,
                  Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor output,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
