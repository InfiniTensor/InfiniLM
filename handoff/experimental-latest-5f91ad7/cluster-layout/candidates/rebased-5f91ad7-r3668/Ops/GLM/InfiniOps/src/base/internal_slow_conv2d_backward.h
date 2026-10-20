#ifndef INFINI_OPS_BASE_INTERNAL_SLOW_CONV2D_BACKWARD_H_
#define INFINI_OPS_BASE_INTERNAL_SLOW_CONV2D_BACKWARD_H_

#include <vector>

#include "operator.h"

namespace infini::ops::internal {

class SlowConv2dBackward : public Operator<SlowConv2dBackward> {
 public:
  SlowConv2dBackward(const Tensor grad_output, const Tensor input,
                     const Tensor weight,
                     const std::vector<int64_t> kernel_size,
                     const std::vector<int64_t> stride,
                     const std::vector<int64_t> padding, Tensor grad_input,
                     Tensor grad_weight, Tensor grad_bias)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        grad_weight_shape_{grad_weight.shape()},
        grad_weight_strides_{grad_weight.strides()},
        grad_weight_type_{grad_weight.dtype()},
        grad_bias_shape_{grad_bias.shape()},
        grad_bias_strides_{grad_bias.strides()},
        grad_bias_type_{grad_bias.dtype()},
        kernel_size_{kernel_size},
        stride_{stride},
        padding_{padding},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const Tensor weight,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding, Tensor grad_input,
                          Tensor grad_weight, Tensor grad_bias) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  Tensor::Shape grad_weight_shape_;

  Tensor::Strides grad_weight_strides_;

  DataType grad_weight_type_;

  Tensor::Shape grad_bias_shape_;

  Tensor::Strides grad_bias_strides_;

  DataType grad_bias_type_;

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> stride_{};

  std::vector<int64_t> padding_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
