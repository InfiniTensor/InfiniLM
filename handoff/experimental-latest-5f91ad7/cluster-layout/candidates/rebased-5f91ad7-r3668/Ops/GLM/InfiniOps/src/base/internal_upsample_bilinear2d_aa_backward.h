#ifndef INFINI_OPS_BASE_INTERNAL_UPSAMPLE_BILINEAR2D_AA_BACKWARD_H_
#define INFINI_OPS_BASE_INTERNAL_UPSAMPLE_BILINEAR2D_AA_BACKWARD_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops::internal {

class UpsampleBilinear2dAaBackward
    : public Operator<UpsampleBilinear2dAaBackward> {
 public:
  UpsampleBilinear2dAaBackward(const Tensor grad_output,
                               const std::vector<int64_t> output_size,
                               const std::vector<int64_t> input_size,
                               const bool align_corners,
                               const std::optional<double> scales_h,
                               const std::optional<double> scales_w,
                               Tensor grad_input)
      : grad_output_shape_{grad_output.shape()},
        grad_output_strides_{grad_output.strides()},
        grad_output_type_{grad_output.dtype()},
        grad_input_shape_{grad_input.shape()},
        grad_input_strides_{grad_input.strides()},
        grad_input_type_{grad_input.dtype()},
        output_size_{output_size},
        input_size_{input_size},
        align_corners_{align_corners},
        scales_h_{scales_h},
        scales_w_{scales_w},
        device_index_{grad_input.device().index()} {}

  virtual void operator()(const Tensor grad_output,
                          const std::vector<int64_t> output_size,
                          const std::vector<int64_t> input_size,
                          const bool align_corners,
                          const std::optional<double> scales_h,
                          const std::optional<double> scales_w,
                          Tensor grad_input) const = 0;

 protected:
  Tensor::Shape grad_output_shape_;

  Tensor::Strides grad_output_strides_;

  DataType grad_output_type_;

  Tensor::Shape grad_input_shape_;

  Tensor::Strides grad_input_strides_;

  DataType grad_input_type_;

  std::vector<int64_t> output_size_{};

  std::vector<int64_t> input_size_{};

  bool align_corners_{};

  std::optional<double> scales_h_{};

  std::optional<double> scales_w_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
