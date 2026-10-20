#ifndef INFINI_OPS_BASE_LOGIT_H_
#define INFINI_OPS_BASE_LOGIT_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Logit : public Operator<Logit> {
 public:
  Logit(const Tensor input, const std::optional<double> eps, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        eps_{eps},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const std::optional<double> eps,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<double> eps_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
