#ifndef INFINI_OPS_BASE_LINALG_LDL_SOLVE_H_
#define INFINI_OPS_BASE_LINALG_LDL_SOLVE_H_

#include "operator.h"

namespace infini::ops::linalg {

class LdlSolve : public Operator<LdlSolve> {
 public:
  LdlSolve(const Tensor LD, const Tensor pivots, const Tensor B,
           const bool hermitian, Tensor out)
      : LD_shape_{LD.shape()},
        LD_strides_{LD.strides()},
        LD_type_{LD.dtype()},
        pivots_shape_{pivots.shape()},
        pivots_strides_{pivots.strides()},
        pivots_type_{pivots.dtype()},
        B_shape_{B.shape()},
        B_strides_{B.strides()},
        B_type_{B.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        hermitian_{hermitian},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor LD, const Tensor pivots, const Tensor B,
                          const bool hermitian, Tensor out) const = 0;

 protected:
  Tensor::Shape LD_shape_;

  Tensor::Strides LD_strides_;

  DataType LD_type_;

  Tensor::Shape pivots_shape_;

  Tensor::Strides pivots_strides_;

  DataType pivots_type_;

  Tensor::Shape B_shape_;

  Tensor::Strides B_strides_;

  DataType B_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool hermitian_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
