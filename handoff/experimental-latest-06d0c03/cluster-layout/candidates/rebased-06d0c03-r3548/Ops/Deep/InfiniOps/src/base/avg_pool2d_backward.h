#ifndef INFINI_OPS_BASE_AVG_POOL2D_BACKWARD_H_
#define INFINI_OPS_BASE_AVG_POOL2D_BACKWARD_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

class AvgPool2dBackward : public Operator<AvgPool2dBackward> {
 public:
  AvgPool2dBackward(const Tensor grad_output, const Tensor input,
                    const std::vector<int64_t> kernel_size,
                    const std::vector<int64_t> stride,
                    const std::vector<int64_t> padding, const bool ceil_mode,
                    const bool count_include_pad,
                    const std::optional<int64_t> divisor_override,
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
        kernel_size_{kernel_size},
        stride_{stride},
        padding_{padding},
        ceil_mode_{ceil_mode},
        count_include_pad_{count_include_pad},
        divisor_override_{divisor_override},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output, const Tensor input,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          const bool ceil_mode, const bool count_include_pad,
                          const std::optional<int64_t> divisor_override,
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

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> stride_{};

  std::vector<int64_t> padding_{};

  bool ceil_mode_{};

  bool count_include_pad_{};

  std::optional<int64_t> divisor_override_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
