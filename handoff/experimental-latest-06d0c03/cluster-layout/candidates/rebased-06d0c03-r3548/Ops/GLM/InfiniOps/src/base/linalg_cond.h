#ifndef INFINI_OPS_BASE_LINALG_COND_H_
#define INFINI_OPS_BASE_LINALG_COND_H_

#include <optional>
#include <string>

#include "operator.h"

namespace infini::ops::linalg {

class Cond : public Operator<Cond> {
 public:
  Cond(const Tensor input, const std::optional<double> p, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        p_{p},
        device_index_{out.device().index()} {}

  Cond(const Tensor input, const std::string p, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const std::optional<double> p,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const std::string p,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<double> p_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
