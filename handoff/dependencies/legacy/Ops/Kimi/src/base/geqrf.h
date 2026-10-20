#ifndef INFINI_OPS_BASE_GEQRF_H_
#define INFINI_OPS_BASE_GEQRF_H_

#include "operator.h"

namespace infini::ops {

class Geqrf : public Operator<Geqrf> {
 public:
  Geqrf(const Tensor input, Tensor a, Tensor tau)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        a_shape_{a.shape()},
        a_strides_{a.strides()},
        a_type_{a.dtype()},
        tau_shape_{tau.shape()},
        tau_strides_{tau.strides()},
        tau_type_{tau.dtype()},
        device_index_{a.device().index()} {}

  virtual void operator()(const Tensor input, Tensor a, Tensor tau) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape a_shape_;

  Tensor::Strides a_strides_;

  DataType a_type_;

  Tensor::Shape tau_shape_;

  Tensor::Strides tau_strides_;

  DataType tau_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
