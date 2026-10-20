#ifndef INFINI_OPS_BASE_INTERNAL_LINALG_SLOGDET_H_
#define INFINI_OPS_BASE_INTERNAL_LINALG_SLOGDET_H_

#include "operator.h"

namespace infini::ops::internal::linalg {

class Slogdet : public Operator<Slogdet> {
 public:
  Slogdet(const Tensor A, Tensor sign, Tensor logabsdet, Tensor LU,
          Tensor pivots)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        sign_shape_{sign.shape()},
        sign_strides_{sign.strides()},
        sign_type_{sign.dtype()},
        logabsdet_shape_{logabsdet.shape()},
        logabsdet_strides_{logabsdet.strides()},
        logabsdet_type_{logabsdet.dtype()},
        LU_shape_{LU.shape()},
        LU_strides_{LU.strides()},
        LU_type_{LU.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        device_index_{sign.device().index()} {}

  virtual void operator()(const Tensor A, Tensor sign, Tensor logabsdet,
                          Tensor LU, Tensor pivots) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape sign_shape_;

  Tensor::Strides sign_strides_;

  DataType sign_type_;

  Tensor::Shape logabsdet_shape_;

  Tensor::Strides logabsdet_strides_;

  DataType logabsdet_type_;

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
