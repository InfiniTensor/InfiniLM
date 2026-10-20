#ifndef INFINI_OPS_BASE_POLAR_H_
#define INFINI_OPS_BASE_POLAR_H_

#include "operator.h"

namespace infini::ops {

class Polar : public Operator<Polar> {
 public:
  Polar(const Tensor abs, const Tensor angle, Tensor out)
      : abs_shape_{abs.shape()},
        abs_strides_{abs.strides()},
        abs_type_{abs.dtype()},
        angle_shape_{angle.shape()},
        angle_strides_{angle.strides()},
        angle_type_{angle.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor abs, const Tensor angle,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape abs_shape_;

  Tensor::Strides abs_strides_;

  DataType abs_type_;

  Tensor::Shape angle_shape_;

  Tensor::Strides angle_strides_;

  DataType angle_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
