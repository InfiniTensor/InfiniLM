#ifndef INFINI_OPS_BASE_LINALG_LU_FACTOR_EX_H_
#define INFINI_OPS_BASE_LINALG_LU_FACTOR_EX_H_

#include "operator.h"

namespace infini::ops::linalg {

class LuFactorEx : public Operator<LuFactorEx> {
 public:
  LuFactorEx(const Tensor A, const bool pivot, const bool check_errors,
             Tensor LU, Tensor pivots, Tensor info)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        LU_shape_{LU.shape()},
        LU_strides_{LU.strides()},
        LU_type_{LU.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        info_shape_{info.shape()},
        info_strides_{info.strides()},
        info_type_{info.dtype()},
        pivot_{pivot},
        check_errors_{check_errors},
        device_index_{LU.device().index()} {}

  virtual void operator()(const Tensor A, const bool pivot,
                          const bool check_errors, Tensor LU, Tensor pivots,
                          Tensor info) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape LU_shape_;

  Tensor::Strides LU_strides_;

  DataType LU_type_;

  Tensor::Shape pivots_shape_;

  Tensor::Strides pivots_strides_;

  DataType pivots_type_;

  Tensor::Shape info_shape_;

  Tensor::Strides info_strides_;

  DataType info_type_;

  bool pivot_{};

  bool check_errors_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
