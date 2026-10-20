#ifndef INFINI_OPS_BASE_SOFTPLUS_H_
#define INFINI_OPS_BASE_SOFTPLUS_H_

#include "operator.h"

namespace infini::ops {

class Softplus : public Operator<Softplus> {
 public:
  Softplus(const Tensor input, const double beta, const double threshold,
           Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        beta_{beta},
        threshold_{threshold},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const double beta,
                          const double threshold, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double beta_{};

  double threshold_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
