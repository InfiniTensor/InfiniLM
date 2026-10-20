#ifndef INFINI_OPS_BASE_LINALG_CROSS_H_
#define INFINI_OPS_BASE_LINALG_CROSS_H_

#include "operator.h"

namespace infini::ops::linalg {

class Cross : public Operator<Cross> {
 public:
  Cross(const Tensor input, const Tensor other, const int64_t dim, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor other,
                          const int64_t dim, Tensor out) const = 0;

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

  int64_t dim_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
