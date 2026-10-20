#ifndef INFINI_OPS_BASE_FRACTIONAL_MAX_POOL2D_BACKWARD_H_
#define INFINI_OPS_BASE_FRACTIONAL_MAX_POOL2D_BACKWARD_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class FractionalMaxPool2dBackward
    : public Operator<FractionalMaxPool2dBackward> {
 public:
  FractionalMaxPool2dBackward(const Tensor grad_output, const Tensor input,
                              const Tensor indices,
                              const std::vector<int64_t> kernel_size,
                              const std::vector<int64_t> output_size,
                              Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        kernel_size_{kernel_size},
        output_size_{output_size},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const Tensor indices,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> output_size,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape indices_shape_;

  Tensor::Strides indices_strides_;

  DataType indices_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> output_size_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
