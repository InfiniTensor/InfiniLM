#ifndef INFINI_OPS_BASE_COMPLEX_H_
#define INFINI_OPS_BASE_COMPLEX_H_

#include "operator.h"

namespace infini::ops {

class Complex : public Operator<Complex> {
 public:
  Complex(const Tensor real, const Tensor imag, Tensor out)
      : real_shape_{real.shape()},
        real_strides_{real.strides()},
        real_type_{real.dtype()},
        imag_shape_{imag.shape()},
        imag_strides_{imag.strides()},
        imag_type_{imag.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor real, const Tensor imag,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape real_shape_;

  Tensor::Strides real_strides_;

  DataType real_type_;

  Tensor::Shape imag_shape_;

  Tensor::Strides imag_strides_;

  DataType imag_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
