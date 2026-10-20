#ifndef INFINI_OPS_BASE_LINALG_CHOLESKY_EX_H_
#define INFINI_OPS_BASE_LINALG_CHOLESKY_EX_H_

#include "operator.h"

namespace infini::ops::linalg {

class CholeskyEx : public Operator<CholeskyEx> {
 public:
  CholeskyEx(const Tensor input, const bool upper, const bool check_errors,
             Tensor L, Tensor info)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        L_shape_{L.shape()},
        L_strides_{L.strides()},
        L_type_{L.dtype()},
        info_shape_{info.shape()},
        info_strides_{info.strides()},
        info_type_{info.dtype()},
        upper_{upper},
        check_errors_{check_errors},
        device_index_{L.device().index()} {}

  virtual void operator()(const Tensor input, const bool upper,
                          const bool check_errors, Tensor L,
                          Tensor info) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape L_shape_;

  Tensor::Strides L_strides_;

  DataType L_type_;

  Tensor::Shape info_shape_;

  Tensor::Strides info_strides_;

  DataType info_type_;

  bool upper_{};

  bool check_errors_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
