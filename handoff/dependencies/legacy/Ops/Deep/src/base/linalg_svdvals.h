#ifndef INFINI_OPS_BASE_LINALG_SVDVALS_H_
#define INFINI_OPS_BASE_LINALG_SVDVALS_H_

#include <optional>
#include <string>

#include "operator.h"

namespace infini::ops::linalg {

class Svdvals : public Operator<Svdvals> {
 public:
  Svdvals(const Tensor A, const std::optional<std::string> driver, Tensor out)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        driver_{driver},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor A,
                          const std::optional<std::string> driver,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<std::string> driver_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
