#ifndef INFINI_OPS_BASE_LINALG_EIGH_H_
#define INFINI_OPS_BASE_LINALG_EIGH_H_

#include <string>

#include "operator.h"

namespace infini::ops::linalg {

class Eigh : public Operator<Eigh> {
 public:
  Eigh(const Tensor input, const std::string UPLO, Tensor eigvals,
       Tensor eigvecs)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        eigvals_shape_{eigvals.shape()},
        eigvals_strides_{eigvals.strides()},
        eigvals_type_{eigvals.dtype()},
        eigvecs_shape_{eigvecs.shape()},
        eigvecs_strides_{eigvecs.strides()},
        eigvecs_type_{eigvecs.dtype()},
        UPLO_{UPLO},
        device_index_{eigvals.device().index()} {}

  virtual void operator()(const Tensor input, const std::string UPLO,
                          Tensor eigvals, Tensor eigvecs) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape eigvals_shape_;

  Tensor::Strides eigvals_strides_;

  DataType eigvals_type_;

  Tensor::Shape eigvecs_shape_;

  Tensor::Strides eigvecs_strides_;

  DataType eigvecs_type_;

  std::string UPLO_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
