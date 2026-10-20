#ifndef INFINI_OPS_BASE_LINALG_DET_H_
#define INFINI_OPS_BASE_LINALG_DET_H_

#include "operator.h"

namespace infini::ops::linalg {

class Det : public Operator<Det> {
 public:
  Det(const Tensor A, Tensor out)
      : A_shape_{A.shape()},
        A_strides_{A.strides()},
        A_type_{A.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor A, Tensor out) const = 0;

 protected:
  Tensor::Shape A_shape_;

  Tensor::Strides A_strides_;

  DataType A_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops::linalg

#endif
