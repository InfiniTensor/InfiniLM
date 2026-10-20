#ifndef INFINI_OPS_BASE_NORMAL_H_
#define INFINI_OPS_BASE_NORMAL_H_

#include "operator.h"

namespace infini::ops {

class Normal : public Operator<Normal> {
 public:
  Normal(const Tensor mean, const double std, Tensor out)
      : mean_shape_{mean.shape()},
        mean_strides_{mean.strides()},
        mean_type_{mean.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        std_{std},
        device_index_{out.device().index()} {}

  Normal(const Tensor mean, const Tensor std, Tensor out)
      : mean_shape_{mean.shape()},
        mean_strides_{mean.strides()},
        mean_type_{mean.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        std_shape_{std.shape()},
        std_strides_{std.strides()},
        std_type_{std.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor mean, const double std,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor mean, const Tensor std,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape mean_shape_;

  Tensor::Strides mean_strides_;

  DataType mean_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double std_{};

  Tensor::Shape std_shape_;

  Tensor::Strides std_strides_;

  DataType std_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
