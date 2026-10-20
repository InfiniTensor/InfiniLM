#ifndef INFINI_OPS_BASE_MAX_POOL2D_WITH_INDICES_H_
#define INFINI_OPS_BASE_MAX_POOL2D_WITH_INDICES_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class MaxPool2dWithIndices : public Operator<MaxPool2dWithIndices> {
 public:
  MaxPool2dWithIndices(const Tensor input,
                       const std::vector<int64_t> kernel_size,
                       const std::vector<int64_t> stride,
                       const std::vector<int64_t> padding,
                       const std::vector<int64_t> dilation,
                       const bool ceil_mode, Tensor out, Tensor indices)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        kernel_size_{kernel_size},
        stride_{stride},
        padding_{padding},
        dilation_{dilation},
        ceil_mode_{ceil_mode},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> dilation,
                          const bool ceil_mode, Tensor out,
                          Tensor indices) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  Tensor::Shape indices_shape_;

  Tensor::Strides indices_strides_;

  DataType indices_type_;

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> stride_{};

  std::vector<int64_t> padding_{};

  std::vector<int64_t> dilation_{};

  bool ceil_mode_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
