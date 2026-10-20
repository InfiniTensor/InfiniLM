#ifndef INFINI_OPS_BASE_RRELU_WITH_NOISE_H_
#define INFINI_OPS_BASE_RRELU_WITH_NOISE_H_

#include "operator.h"

namespace infini::ops {

class RreluWithNoise : public Operator<RreluWithNoise> {
 public:
  RreluWithNoise(const Tensor input, const double lower, const double upper,
                 const bool training, Tensor out, Tensor noise)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        noise_shape_{noise.shape()},
        noise_strides_{noise.strides()},
        noise_type_{noise.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        lower_{lower},
        upper_{upper},
        training_{training},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const double lower,
                          const double upper, const bool training, Tensor out,
                          Tensor noise) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape noise_shape_;

  Tensor::Strides noise_strides_;

  DataType noise_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double lower_{};

  double upper_{};

  bool training_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
