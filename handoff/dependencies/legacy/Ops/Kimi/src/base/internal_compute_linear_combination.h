#ifndef INFINI_OPS_BASE_INTERNAL_COMPUTE_LINEAR_COMBINATION_H_
#define INFINI_OPS_BASE_INTERNAL_COMPUTE_LINEAR_COMBINATION_H_

#include "operator.h"

namespace infini::ops::internal {

class ComputeLinearCombination : public Operator<ComputeLinearCombination> {
 public:
  ComputeLinearCombination(const Tensor input, const Tensor coefficients,
                           Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        coefficients_shape_{coefficients.shape()},
        coefficients_strides_{coefficients.strides()},
        coefficients_type_{coefficients.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor coefficients,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape coefficients_shape_;

  Tensor::Strides coefficients_strides_;

  DataType coefficients_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops::internal

#endif
