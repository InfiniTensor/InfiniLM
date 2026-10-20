#ifndef INFINI_OPS_BASE_QR_H_
#define INFINI_OPS_BASE_QR_H_

#include "operator.h"

namespace infini::ops {

class Qr : public Operator<Qr> {
 public:
  Qr(const Tensor input, const bool some, Tensor Q, Tensor R)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        Q_shape_{Q.shape()},
        Q_strides_{Q.strides()},
        Q_type_{Q.dtype()},
        R_shape_{R.shape()},
        R_strides_{R.strides()},
        R_type_{R.dtype()},
        some_{some},
        device_index_{Q.device().index()} {}

  virtual void operator()(const Tensor input, const bool some, Tensor Q,
                          Tensor R) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape Q_shape_;

  Tensor::Strides Q_strides_;

  DataType Q_type_;

  Tensor::Shape R_shape_;

  Tensor::Strides R_strides_;

  DataType R_type_;

  bool some_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
