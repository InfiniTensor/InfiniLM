#ifndef INFINI_OPS_BASE_SLOGDET_H_
#define INFINI_OPS_BASE_SLOGDET_H_

#include "operator.h"

namespace infini::ops {

class Slogdet : public Operator<Slogdet> {
 public:
  Slogdet(const Tensor input, Tensor sign, Tensor logabsdet)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        sign_shape_{sign.shape()},
        sign_strides_{sign.strides()},
        sign_type_{sign.dtype()},
        logabsdet_shape_{logabsdet.shape()},
        logabsdet_strides_{logabsdet.strides()},
        logabsdet_type_{logabsdet.dtype()},
        device_index_{sign.device().index()} {}

  virtual void operator()(const Tensor input, Tensor sign,
                          Tensor logabsdet) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape sign_shape_;

  Tensor::Strides sign_strides_;

  DataType sign_type_;

  Tensor::Shape logabsdet_shape_;

  Tensor::Strides logabsdet_strides_;

  DataType logabsdet_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
