#ifndef INFINI_OPS_BASE_INTERNAL_LINALG_DET_H_
#define INFINI_OPS_BASE_INTERNAL_LINALG_DET_H_

#include "operator.h"

namespace infini::ops::internal::linalg {

class Det : public Operator<Det> {
 public:
  Det(const Tensor A, Tensor result, Tensor LU, Tensor pivots)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        result_shape_{result.shape()},
        result_strides_{result.strides()},
        result_type_{result.dtype()},
        LU_shape_{LU.shape()},
        LU_strides_{LU.strides()},
        LU_type_{LU.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        device_index_{result.device().index()} {}

  virtual void operator()(const Tensor A, Tensor result, Tensor LU,
                          Tensor pivots) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape result_shape_;

  Tensor::Strides result_strides_;

  DataType result_type_;

  Tensor::Shape LU_shape_;

  Tensor::Strides LU_strides_;

  DataType LU_type_;

  Tensor::Shape pivots_shape_;

  Tensor::Strides pivots_strides_;

  DataType pivots_type_;

  int device_index_{0};
};

}  // namespace infini::ops::internal::linalg

#endif
