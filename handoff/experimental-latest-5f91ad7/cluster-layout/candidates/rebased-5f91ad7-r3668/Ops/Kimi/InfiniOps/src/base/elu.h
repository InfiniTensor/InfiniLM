#ifndef INFINI_OPS_BASE_ELU_H_
#define INFINI_OPS_BASE_ELU_H_

#include "operator.h"

namespace infini::ops {

class Elu : public Operator<Elu> {
 public:
  Elu(const Tensor input, const double alpha, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        alpha_{alpha},
        scale_{1.0},
        input_scale_{1.0},
        device_index_{out.device().index()} {}

  /// \deprecated Use `Elu(input, alpha, out)`. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the `(input, alpha, out)` overload instead.")]]
  Elu(const Tensor input, const double alpha, const double scale,
      const double input_scale, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        alpha_{alpha},
        scale_{scale},
        input_scale_{input_scale},
        device_index_{out.device().index()} {}

  void operator()(const Tensor input, const double alpha, Tensor out) const {
    return operator()(input, alpha, 1.0, 1.0, out);
  }

  /// \deprecated Use `operator()(input, alpha, out)`. This overload will be
  /// removed in a future release.
  [[deprecated("Use `operator()(input, alpha, out)` instead.")]]
  virtual void operator()(const Tensor input, const double alpha,
                          const double scale, const double input_scale,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double alpha_{};

  double scale_{};

  double input_scale_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
