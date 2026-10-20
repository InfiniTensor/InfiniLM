#ifndef INFINI_OPS_BASE_SPECIAL_MULTIGAMMALN_H_
#define INFINI_OPS_BASE_SPECIAL_MULTIGAMMALN_H_

#include "operator.h"

namespace infini::ops::special {

class Multigammaln : public Operator<Multigammaln> {
 public:
  Multigammaln(const Tensor input, const int64_t p, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        p_{p},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t p,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t p_{};

  int device_index_{0};
};

}  // namespace infini::ops::special

#endif
