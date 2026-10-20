#ifndef INFINI_OPS_BASE_TRIANGULAR_SOLVE_H_
#define INFINI_OPS_BASE_TRIANGULAR_SOLVE_H_

#include "operator.h"

namespace infini::ops {

class TriangularSolve : public Operator<TriangularSolve> {
 public:
  TriangularSolve(const Tensor input, const Tensor A, const bool upper,
                  const bool transpose, const bool unitriangular, Tensor X,
                  Tensor M)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        X_shape_{X.shape()},
        X_strides_{X.strides()},
        X_type_{X.dtype()},
        M_shape_{M.shape()},
        M_strides_{M.strides()},
        M_type_{M.dtype()},
        upper_{upper},
        transpose_{transpose},
        unitriangular_{unitriangular},
        device_index_{X.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor A, const bool upper,
                          const bool transpose, const bool unitriangular,
                          Tensor X, Tensor M) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape X_shape_;

  Tensor::Strides X_strides_;

  DataType X_type_;

  Tensor::Shape M_shape_;

  Tensor::Strides M_strides_;

  DataType M_type_;

  bool upper_{};

  bool transpose_{};

  bool unitriangular_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
