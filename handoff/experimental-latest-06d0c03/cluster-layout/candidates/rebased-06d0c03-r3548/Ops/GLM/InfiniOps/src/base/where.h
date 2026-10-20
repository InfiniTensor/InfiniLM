#ifndef INFINI_OPS_BASE_WHERE_H_
#define INFINI_OPS_BASE_WHERE_H_

#include "operator.h"

namespace infini::ops {

class Where : public Operator<Where> {
 public:
  Where(const Tensor condition, const Tensor input, const Tensor other,
        Tensor out)
      : condition_shape_{condition.shape()},
        condition_strides_{condition.strides()},
        condition_type_{condition.dtype()},
        input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor condition, const Tensor input,
                          const Tensor other, Tensor out) const = 0;

 protected:
  Tensor::Shape condition_shape_;

  Tensor::Strides condition_strides_;

  DataType condition_type_;

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
