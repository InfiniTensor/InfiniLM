#ifndef INFINI_OPS_BASE_LINALG_SOLVE_H_
#define INFINI_OPS_BASE_LINALG_SOLVE_H_

#include "operator.h"

namespace infini::ops::linalg {

class Solve : public Operator<Solve> {
 public:
  Solve(const Tensor A, const Tensor B, const bool left, Tensor out)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        B_shape_{B.shape()},
        B_strides_{B.strides()},
        B_type_{B.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        left_{left},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor A, const Tensor B, const bool left,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape B_shape_;

  Tensor::Strides B_strides_;

  DataType B_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool left_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
