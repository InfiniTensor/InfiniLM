#ifndef INFINI_OPS_BASE_LINALG_TENSORINV_H_
#define INFINI_OPS_BASE_LINALG_TENSORINV_H_

#include "operator.h"

namespace infini::ops::linalg {

class Tensorinv : public Operator<Tensorinv> {
 public:
  Tensorinv(const Tensor input, const int64_t ind, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        ind_{ind},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const int64_t ind,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t ind_{};

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
