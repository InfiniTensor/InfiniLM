#ifndef INFINI_OPS_BASE_AVG_POOL3D_H_
#define INFINI_OPS_BASE_AVG_POOL3D_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

class AvgPool3d : public Operator<AvgPool3d> {
 public:
  AvgPool3d(const Tensor input, const std::vector<int64_t> kernel_size,
            const std::vector<int64_t> stride,
            const std::vector<int64_t> padding, const bool ceil_mode,
            const bool count_include_pad,
            const std::optional<int64_t> divisor_override, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        kernel_size_{kernel_size},
        stride_{stride},
        padding_{padding},
        ceil_mode_{ceil_mode},
        count_include_pad_{count_include_pad},
        divisor_override_{divisor_override},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          const bool ceil_mode, const bool count_include_pad,
                          const std::optional<int64_t> divisor_override,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

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
