#ifndef INFINI_OPS_BASE_FREXP_H_
#define INFINI_OPS_BASE_FREXP_H_

#include "operator.h"

namespace infini::ops {

class Frexp : public Operator<Frexp> {
 public:
  Frexp(const Tensor input, Tensor mantissa, Tensor exponent)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        mantissa_shape_{mantissa.shape()},
        mantissa_strides_{mantissa.strides()},
        mantissa_type_{mantissa.dtype()},
        exponent_shape_{exponent.shape()},
        exponent_strides_{exponent.strides()},
        exponent_type_{exponent.dtype()},
        device_index_{mantissa.device().index()} {}

  virtual void operator()(const Tensor input, Tensor mantissa,
                          Tensor exponent) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape mantissa_shape_;

  Tensor::Strides mantissa_strides_;

  DataType mantissa_type_;

  Tensor::Shape exponent_shape_;

  Tensor::Strides exponent_strides_;

  DataType exponent_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
