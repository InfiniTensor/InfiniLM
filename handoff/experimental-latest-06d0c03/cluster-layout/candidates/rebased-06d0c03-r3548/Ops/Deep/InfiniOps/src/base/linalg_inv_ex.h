#ifndef INFINI_OPS_BASE_LINALG_INV_EX_H_
#define INFINI_OPS_BASE_LINALG_INV_EX_H_

#include "operator.h"

namespace infini::ops::linalg {

class InvEx : public Operator<InvEx> {
 public:
  InvEx(const Tensor A, const bool check_errors, Tensor inverse, Tensor info)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        inverse_shape_{inverse.shape()},
        inverse_strides_{inverse.strides()},
        inverse_type_{inverse.dtype()},
        info_shape_{info.shape()},
        info_strides_{info.strides()},
        info_type_{info.dtype()},
        check_errors_{check_errors},
        device_index_{inverse.device().index()} {}

  virtual void operator()(const Tensor A, const bool check_errors,
                          Tensor inverse, Tensor info) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape inverse_shape_;

  Tensor::Strides inverse_strides_;

  DataType inverse_type_;

  Tensor::Shape info_shape_;

  Tensor::Strides info_strides_;

  DataType info_type_;

  bool check_errors_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
