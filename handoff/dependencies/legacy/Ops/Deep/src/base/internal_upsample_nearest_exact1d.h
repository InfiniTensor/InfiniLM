#ifndef INFINI_OPS_BASE_INTERNAL_UPSAMPLE_NEAREST_EXACT1D_H_
#define INFINI_OPS_BASE_INTERNAL_UPSAMPLE_NEAREST_EXACT1D_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops::internal {

class UpsampleNearestExact1d : public Operator<UpsampleNearestExact1d> {
 public:
  UpsampleNearestExact1d(const Tensor input,
                         const std::vector<int64_t> output_size,
                         const std::optional<double> scales, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        output_size_{output_size},
        scales_{scales},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> output_size,
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

  std::optional<double> scales_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
