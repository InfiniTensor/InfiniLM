#ifndef INFINI_OPS_BASE_LINALG_SOLVE_TRIANGULAR_H_
#define INFINI_OPS_BASE_LINALG_SOLVE_TRIANGULAR_H_

#include "operator.h"

namespace infini::ops::linalg {

class SolveTriangular : public Operator<SolveTriangular> {
 public:
  SolveTriangular(const Tensor input, const Tensor B, const bool upper,
                  const bool left, const bool unitriangular, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        B_shape_{B.shape()},
        B_strides_{B.strides()},
        B_type_{B.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        upper_{upper},
        left_{left},
        unitriangular_{unitriangular},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor B, const bool upper,
                          const bool left, const bool unitriangular,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape B_shape_;

  Tensor::Strides B_strides_;

  DataType B_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  bool upper_{};

  bool left_{};

  bool unitriangular_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
