#ifndef INFINI_OPS_BASE_INTERNAL_LINALG_EIGH_H_
#define INFINI_OPS_BASE_INTERNAL_LINALG_EIGH_H_

#include <string>

#include "operator.h"

namespace infini::ops::internal::linalg {

class Eigh : public Operator<Eigh> {
 public:
  Eigh(const Tensor A, const std::string UPLO, const bool compute_v,
       Tensor eigenvalues, Tensor eigenvectors)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        eigenvalues_shape_{eigenvalues.shape()},
        eigenvalues_strides_{eigenvalues.strides()},
        eigenvalues_type_{eigenvalues.dtype()},
        eigenvectors_shape_{eigenvectors.shape()},
        eigenvectors_strides_{eigenvectors.strides()},
        eigenvectors_type_{eigenvectors.dtype()},
        UPLO_{UPLO},
        compute_v_{compute_v},
        device_index_{eigenvalues.device().index()} {}

  virtual void operator()(const Tensor A, const std::string UPLO,
                          const bool compute_v, Tensor eigenvalues,
                          Tensor eigenvectors) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape eigenvalues_shape_;

  Tensor::Strides eigenvalues_strides_;

  DataType eigenvalues_type_;

  Tensor::Shape eigenvectors_shape_;

  Tensor::Strides eigenvectors_strides_;

  DataType eigenvectors_type_;

  std::string UPLO_{};

  bool compute_v_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal::linalg

#endif
