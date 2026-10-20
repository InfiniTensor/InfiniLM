#ifndef INFINI_OPS_BASE_MIN_H_
#define INFINI_OPS_BASE_MIN_H_

#include "operator.h"

namespace infini::ops {

class Min : public Operator<Min> {
 public:
  Min(const Tensor input, const Tensor other, Tensor out)
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

  Min(const Tensor input, const int64_t dim, const bool keepdim, Tensor min,
      Tensor min_indices)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        min_shape_{min.shape()},
        min_strides_{min.strides()},
        min_type_{min.dtype()},
        min_indices_shape_{min_indices.shape()},
        min_indices_strides_{min_indices.strides()},
        min_indices_type_{min_indices.dtype()},
        dim_{dim},
        keepdim_{keepdim},
        device_index_{min.device().index()} {}

  Min(const Tensor input, Tensor out)
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
                          const bool keepdim, Tensor min,
                          Tensor min_indices) const = 0;

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

  Tensor::Shape min_shape_;

  Tensor::Strides min_strides_;

  DataType min_type_;

  Tensor::Shape min_indices_shape_;

  Tensor::Strides min_indices_strides_;

  DataType min_indices_type_;

  int64_t dim_{};

  bool keepdim_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
