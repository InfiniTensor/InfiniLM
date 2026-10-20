#ifndef INFINI_OPS_BASE_LEAKY_RELU_BACKWARD_H_
#define INFINI_OPS_BASE_LEAKY_RELU_BACKWARD_H_

#include "operator.h"

namespace infini::ops {

class LeakyReluBackward : public Operator<LeakyReluBackward> {
 public:
  LeakyReluBackward(const Tensor grad_output, const Tensor input,
                    const double negative_slope, const bool self_is_result,
                    Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        negative_slope_{negative_slope},
        self_is_result_{self_is_result},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const double negative_slope,
                          const bool self_is_result,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  double negative_slope_{};

  bool self_is_result_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
