#ifndef INFINI_OPS_BASE_MAX_H_
#define INFINI_OPS_BASE_MAX_H_

#include "operator.h"

namespace infini::ops {

class Max : public Operator<Max> {
 public:
  Max(const Tensor input, const Tensor other, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  Max(const Tensor input, const int64_t dim, const bool keepdim, Tensor max,
      Tensor max_values)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        max_values_shape_{max_values.shape()},
        max_values_strides_{max_values.strides()},
        max_values_type_{max_values.dtype()},
        dim_{dim},
        keepdim_{keepdim},
        device_index_{max.device().index()} {}

  Max(const Tensor input, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor other,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const int64_t dim,
                          const bool keepdim, Tensor max,
                          Tensor max_values) const = 0;

  virtual void operator()(const Tensor input, Tensor out) const = 0;

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

  Tensor::Shape max_shape_;

  Tensor::Strides max_strides_;

  DataType max_type_;

  Tensor::Shape max_values_shape_;

  Tensor::Strides max_values_strides_;

  DataType max_values_type_;

  int64_t dim_{};

  bool keepdim_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
