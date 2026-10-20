#ifndef INFINI_OPS_BASE_LINALG_LU_SOLVE_H_
#define INFINI_OPS_BASE_LINALG_LU_SOLVE_H_

#include "operator.h"

namespace infini::ops::linalg {

class LuSolve : public Operator<LuSolve> {
 public:
  LuSolve(const Tensor LU, const Tensor pivots, const Tensor B, const bool left,
          const bool adjoint, Tensor out)
      : LU_shape_{LU.shape()},
        LU_strides_{LU.strides()},
        LU_type_{LU.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        B_shape_{B.shape()},
        B_strides_{B.strides()},
        B_type_{B.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        left_{left},
        adjoint_{adjoint},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor LU, const Tensor pivots, const Tensor B,
                          const bool left, const bool adjoint,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape LU_shape_;

  Tensor::Strides LU_strides_;

  DataType LU_type_;

  Tensor::Shape pivots_shape_;

  Tensor::Strides pivots_strides_;

  DataType pivots_type_;

  Tensor::Shape B_shape_;

  Tensor::Strides B_strides_;

  DataType B_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool left_{};

  bool adjoint_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
