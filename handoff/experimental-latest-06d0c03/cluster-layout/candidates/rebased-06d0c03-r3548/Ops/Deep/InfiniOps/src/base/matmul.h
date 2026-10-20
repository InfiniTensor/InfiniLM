#ifndef INFINI_OPS_BASE_MATMUL_H_
#define INFINI_OPS_BASE_MATMUL_H_

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class Matmul : public Operator<Matmul> {
 public:
  Matmul(const Tensor input, const Tensor other, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()} {
    assert(input.dtype() == other.dtype() &&
           "operator `Matmul` requires inputs to have the same dtype");
    assert(input.dtype() == out.dtype() &&
           "operator `Matmul` requires output to have the input dtype");
  }

  virtual void operator()(const Tensor input, const Tensor other,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;
};

}  // namespace infini::ops

#endif
