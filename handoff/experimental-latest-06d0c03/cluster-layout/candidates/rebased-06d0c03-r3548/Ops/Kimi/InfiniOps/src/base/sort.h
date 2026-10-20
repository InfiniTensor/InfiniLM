#ifndef INFINI_OPS_BASE_SORT_H_
#define INFINI_OPS_BASE_SORT_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Sort : public Operator<Sort> {
 public:
  Sort(const Tensor input, const int64_t dim, const bool descending,
       Tensor values, Tensor indices)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        values_shape_{values.shape()},
        values_strides_{values.strides()},
        values_type_{values.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        dim_{dim},
        descending_{descending},
        device_index_{values.device().index()} {}

  Sort(const Tensor input, const std::optional<bool> stable, const int64_t dim,
       const bool descending, Tensor values, Tensor indices)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        values_shape_{values.shape()},
        values_strides_{values.strides()},
        values_type_{values.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        dim_{dim},
        descending_{descending},
        stable_{stable},
        device_index_{values.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t dim,
                          const bool descending, Tensor values,
                          Tensor indices) const = 0;

  virtual void operator()(const Tensor input, const std::optional<bool> stable,
                          const int64_t dim, const bool descending,
                          Tensor values, Tensor indices) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape values_shape_;

  Tensor::Strides values_strides_;

  DataType values_type_;

  Tensor::Shape indices_shape_;

  Tensor::Strides indices_strides_;

  DataType indices_type_;

  int64_t dim_{};

  bool descending_{};

  std::optional<bool> stable_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
