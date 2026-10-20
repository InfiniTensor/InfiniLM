#ifndef INFINI_OPS_BASE_HEAVISIDE_H_
#define INFINI_OPS_BASE_HEAVISIDE_H_

#include "operator.h"

namespace infini::ops {

class Heaviside : public Operator<Heaviside> {
 public:
  Heaviside(const Tensor input, const Tensor values, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        values_shape_{values.shape()},
        values_strides_{values.strides()},
        values_type_{values.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor values,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape values_shape_;

  Tensor::Strides values_strides_;

  DataType values_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
