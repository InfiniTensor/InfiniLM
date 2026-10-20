#ifndef INFINI_OPS_BASE_LINALG_SOLVE_EX_H_
#define INFINI_OPS_BASE_LINALG_SOLVE_EX_H_

#include "operator.h"

namespace infini::ops::linalg {

class SolveEx : public Operator<SolveEx> {
 public:
  SolveEx(const Tensor A, const Tensor B, const bool left,
          const bool check_errors, Tensor result, Tensor info)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        B_shape_{B.shape()},
        B_strides_{B.strides()},
        B_type_{B.dtype()},
        result_shape_{result.shape()},
        result_strides_{result.strides()},
        result_type_{result.dtype()},
        info_shape_{info.shape()},
        info_strides_{info.strides()},
        info_type_{info.dtype()},
        left_{left},
        check_errors_{check_errors},
        device_index_{result.device().index()} {}

  virtual void operator()(const Tensor A, const Tensor B, const bool left,
                          const bool check_errors, Tensor result,
                          Tensor info) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape B_shape_;

  Tensor::Strides B_strides_;

  DataType B_type_;

  Tensor::Shape result_shape_;

  Tensor::Strides result_strides_;

  DataType result_type_;

  Tensor::Shape info_shape_;

  Tensor::Strides info_strides_;

  DataType info_type_;

  bool left_{};

  bool check_errors_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
