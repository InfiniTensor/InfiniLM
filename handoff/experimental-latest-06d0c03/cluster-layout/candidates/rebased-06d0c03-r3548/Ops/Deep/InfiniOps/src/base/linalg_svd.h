#ifndef INFINI_OPS_BASE_LINALG_SVD_H_
#define INFINI_OPS_BASE_LINALG_SVD_H_

#include <optional>
#include <string>

#include "operator.h"

namespace infini::ops::linalg {

class Svd : public Operator<Svd> {
 public:
  Svd(const Tensor A, const bool full_matrices,
      const std::optional<std::string> driver, Tensor U, Tensor S, Tensor Vh)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        U_shape_{U.shape()},
        U_strides_{U.strides()},
        U_type_{U.dtype()},
        S_shape_{S.shape()},
        S_strides_{S.strides()},
        S_type_{S.dtype()},
        Vh_shape_{Vh.shape()},
        Vh_strides_{Vh.strides()},
        Vh_type_{Vh.dtype()},
        full_matrices_{full_matrices},
        driver_{driver},
        device_index_{U.device().index()} {}

  virtual void operator()(const Tensor A, const bool full_matrices,
                          const std::optional<std::string> driver, Tensor U,
                          Tensor S, Tensor Vh) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape U_shape_;

  Tensor::Strides U_strides_;

  DataType U_type_;

  Tensor::Shape S_shape_;

  Tensor::Strides S_strides_;

  DataType S_type_;

  Tensor::Shape Vh_shape_;

  Tensor::Strides Vh_strides_;

  DataType Vh_type_;

  bool full_matrices_{};

  std::optional<std::string> driver_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
