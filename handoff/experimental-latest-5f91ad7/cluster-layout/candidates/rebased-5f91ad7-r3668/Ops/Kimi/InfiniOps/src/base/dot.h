#ifndef INFINI_OPS_BASE_DOT_H_
#define INFINI_OPS_BASE_DOT_H_

#include "operator.h"

namespace infini::ops {

class Dot : public Operator<Dot> {
 public:
  Dot(const Tensor input, const Tensor tensor, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        tensor_shape_{tensor.shape()},
        tensor_strides_{tensor.strides()},
        tensor_type_{tensor.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor tensor,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape tensor_shape_;

  Tensor::Strides tensor_strides_;

  DataType tensor_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
