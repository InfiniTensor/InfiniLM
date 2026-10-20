#ifndef INFINI_OPS_BASE_SUBTRACT_H_
#define INFINI_OPS_BASE_SUBTRACT_H_

#include "operator.h"

namespace infini::ops {

class Subtract : public Operator<Subtract> {
 public:
  Subtract(const Tensor input, const Tensor other, const double alpha,
           Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        alpha_{alpha},
        device_index_{out.device().index()} {}

  /// \deprecated Use the explicit-output constructor. This constructor will
  /// be removed in a future release.
  [[deprecated("Use the `(input, other, alpha, out)` overload instead.")]]
  Subtract(Tensor input, const Tensor other, const double alpha)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        alpha_{alpha},
        device_index_{input.device().index()} {}

  Subtract(const Tensor input, const double other, const double alpha,
           Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        alpha_{alpha},
        other_{other},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor other,
                          const double alpha, Tensor out) const = 0;

  /// \deprecated Use the explicit-output overload. This overload will be
  /// removed in a future release.
  [[deprecated("Use `operator()(input, other, alpha, out)` instead.")]]
  virtual void operator()(Tensor input, const Tensor other,
                          const double alpha) const = 0;

  virtual void operator()(const Tensor input, const double other,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double alpha_{};

  double other_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
