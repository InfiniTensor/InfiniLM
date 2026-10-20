#ifndef INFINI_OPS_BASE_LINALG_EIG_H_
#define INFINI_OPS_BASE_LINALG_EIG_H_

#include "operator.h"

namespace infini::ops::linalg {

class Eig : public Operator<Eig> {
 public:
  Eig(const Tensor input, Tensor eigenvalues, Tensor eigenvectors)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        eigenvalues_shape_{eigenvalues.shape()},
        eigenvalues_strides_{eigenvalues.strides()},
        eigenvalues_type_{eigenvalues.dtype()},
        eigenvectors_shape_{eigenvectors.shape()},
        eigenvectors_strides_{eigenvectors.strides()},
        eigenvectors_type_{eigenvectors.dtype()},
        device_index_{eigenvalues.device().index()} {}

  virtual void operator()(const Tensor input, Tensor eigenvalues,
                          Tensor eigenvectors) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape eigenvalues_shape_;

  Tensor::Strides eigenvalues_strides_;

  DataType eigenvalues_type_;

  Tensor::Shape eigenvectors_shape_;

  Tensor::Strides eigenvectors_strides_;

  DataType eigenvectors_type_;

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
