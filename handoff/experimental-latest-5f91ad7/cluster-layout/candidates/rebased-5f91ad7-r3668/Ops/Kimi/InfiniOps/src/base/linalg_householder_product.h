#ifndef INFINI_OPS_BASE_LINALG_HOUSEHOLDER_PRODUCT_H_
#define INFINI_OPS_BASE_LINALG_HOUSEHOLDER_PRODUCT_H_

#include "operator.h"

namespace infini::ops::linalg {

class HouseholderProduct : public Operator<HouseholderProduct> {
 public:
  HouseholderProduct(const Tensor input, const Tensor tau, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        tau_shape_{tau.shape()},
        tau_strides_{tau.strides()},
        tau_type_{tau.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor tau,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape tau_shape_;

  Tensor::Strides tau_strides_;

  DataType tau_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
