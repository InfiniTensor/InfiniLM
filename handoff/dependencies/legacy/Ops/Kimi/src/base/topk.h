#ifndef INFINI_OPS_BASE_TOPK_H_
#define INFINI_OPS_BASE_TOPK_H_

#include "operator.h"

namespace infini::ops {

class Topk : public Operator<Topk> {
 public:
  Topk(const Tensor input, const int64_t k, const int64_t dim,
       const bool largest, const bool sorted, Tensor values, Tensor indices)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        values_shape_{values.shape()},
        values_strides_{values.strides()},
        values_type_{values.dtype()},
        indices_shape_{indices.shape()},
        indices_strides_{indices.strides()},
        indices_type_{indices.dtype()},
        k_{k},
        dim_{dim},
        largest_{largest},
        sorted_{sorted},
        device_index_{values.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t k,
                          const int64_t dim, const bool largest,
                          const bool sorted, Tensor values,
                          Tensor indices) const = 0;

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

  int64_t k_{};

  int64_t dim_{};

  bool largest_{};

  bool sorted_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
