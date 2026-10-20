#ifndef INFINI_OPS_BASE_UPSAMPLE_LINEAR1D_H_
#define INFINI_OPS_BASE_UPSAMPLE_LINEAR1D_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

class UpsampleLinear1d : public Operator<UpsampleLinear1d> {
 public:
  UpsampleLinear1d(const Tensor input, const std::vector<int64_t> output_size,
                   const bool align_corners, const std::optional<double> scales,
                   Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{output_size},
        align_corners_{align_corners},
        scales_{scales},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> output_size,
                          const bool align_corners,
                          const std::optional<double> scales,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> output_size_{};

  bool align_corners_{};

  std::optional<double> scales_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
