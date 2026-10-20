#ifndef INFINI_OPS_BASE_LINALG_VECDOT_H_
#define INFINI_OPS_BASE_LINALG_VECDOT_H_

#include "operator.h"

namespace infini::ops::linalg {

class Vecdot : public Operator<Vecdot> {
 public:
  Vecdot(const Tensor x, const Tensor y, const int64_t dim, Tensor out)
      : x_shape_{x.shape()},
        x_strides_{x.strides()},
        x_type_{x.dtype()},
        y_shape_{y.shape()},
        y_strides_{y.strides()},
        y_type_{y.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor x, const Tensor y, const int64_t dim,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape x_shape_;

  Tensor::Strides x_strides_;

  DataType x_type_;

  Tensor::Shape y_shape_;

  Tensor::Strides y_strides_;

  DataType y_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t dim_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
