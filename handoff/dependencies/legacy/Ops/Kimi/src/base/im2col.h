#ifndef INFINI_OPS_BASE_IM2COL_H_
#define INFINI_OPS_BASE_IM2COL_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class Im2col : public Operator<Im2col> {
 public:
  Im2col(const Tensor input, const std::vector<int64_t> kernel_size,
         const std::vector<int64_t> dilation,
         const std::vector<int64_t> padding, const std::vector<int64_t> stride,
         Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        kernel_size_{kernel_size},
        dilation_{dilation},
        padding_{padding},
        stride_{stride},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> dilation,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> stride,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> dilation_{};

  std::vector<int64_t> padding_{};

  std::vector<int64_t> stride_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
