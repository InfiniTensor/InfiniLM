#ifndef INFINI_OPS_BASE_RMS_NORM_H_
#define INFINI_OPS_BASE_RMS_NORM_H_

#include <cstddef>
#include <vector>

#include "operator.h"
#include "tensor.h"

namespace infini::ops {

class RmsNorm : public Operator<RmsNorm> {
 public:
  RmsNorm(const Tensor input, const Tensor weight, float eps, Tensor out)
      : input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        eps_{eps},
        dim_{out.size(-1)},
        ndim_{out.ndim()},
        batch_size_{ndim_ == 2 ? out.size(-2) : out.size(-3)},
        nhead_{ndim_ == 2 ? 1 : out.size(-2)} {
    assert(input.dtype() == out.dtype());
  }

  RmsNorm(const Tensor input, const Tensor weight, Tensor out)
      : RmsNorm{input, weight, 1e-6f, out} {}

  // TODO: Type of `eps` should be `std::optional<float>` instead of `float`.
  virtual void operator()(const Tensor input, const Tensor weight, float eps,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor weight,
                          Tensor out) const {
    return operator()(input, weight, eps_, out);
  }

 protected:
  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  float eps_{1e-6f};

  Tensor::Size dim_{0};

  Tensor::Size ndim_{0};

  Tensor::Size batch_size_{0};

  Tensor::Size nhead_{1};
};

}  // namespace infini::ops

#endif
