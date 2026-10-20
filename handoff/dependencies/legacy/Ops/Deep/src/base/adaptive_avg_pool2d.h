#ifndef INFINI_OPS_BASE_ADAPTIVE_AVG_POOL2D_H_
#define INFINI_OPS_BASE_ADAPTIVE_AVG_POOL2D_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class AdaptiveAvgPool2d : public Operator<AdaptiveAvgPool2d> {
 public:
  AdaptiveAvgPool2d(const Tensor input, const std::vector<int64_t> output_size,
                    Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{output_size},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> output_size,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> output_size_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
