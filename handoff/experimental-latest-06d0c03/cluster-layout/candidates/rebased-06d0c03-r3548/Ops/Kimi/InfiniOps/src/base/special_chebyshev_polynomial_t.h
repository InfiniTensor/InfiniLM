#ifndef INFINI_OPS_BASE_SPECIAL_CHEBYSHEV_POLYNOMIAL_T_H_
#define INFINI_OPS_BASE_SPECIAL_CHEBYSHEV_POLYNOMIAL_T_H_

#include "operator.h"

namespace infini::ops::special {

class ChebyshevPolynomialT : public Operator<ChebyshevPolynomialT> {
 public:
  ChebyshevPolynomialT(const Tensor x, const Tensor n, Tensor out)
      : x_shape_{x.shape()},
        x_strides_{x.strides()},
        x_type_{x.dtype()},
        n_shape_{n.shape()},
        n_strides_{n.strides()},
        n_type_{n.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  ChebyshevPolynomialT(const Tensor x, const double n, Tensor out)
      : x_shape_{x.shape()},
        x_strides_{x.strides()},
        x_type_{x.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        n_{n},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor x, const Tensor n, Tensor out) const = 0;

  virtual void operator()(const Tensor x, const double n, Tensor out) const = 0;

 protected:
  Tensor::Shape x_shape_;

  Tensor::Strides x_strides_;

  DataType x_type_;

  Tensor::Shape n_shape_;

  Tensor::Strides n_strides_;

  DataType n_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double n_{};

  int device_index_{0};
};

}  // namespace infini::ops::special

#endif
