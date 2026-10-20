#ifndef INFINI_OPS_BASE_FLOAT_POWER_H_
#define INFINI_OPS_BASE_FLOAT_POWER_H_

#include "operator.h"

namespace infini::ops {

class FloatPower : public Operator<FloatPower> {
 public:
  FloatPower(const Tensor input, const Tensor exponent, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        exponent_shape_{exponent.shape()},
        exponent_strides_{exponent.strides()},
        exponent_type_{exponent.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  FloatPower(const Tensor input, const double exponent, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        exponent_{exponent},
        device_index_{out.device().index()} {}

  /// \deprecated Use the explicit-output constructor. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the explicit-output overload instead.")]]
  FloatPower(Tensor input, const double exponent)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        exponent_{exponent},
        device_index_{input.device().index()} {}

  /// \deprecated Use the explicit-output constructor. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the explicit-output overload instead.")]]
  FloatPower(Tensor input, const Tensor exponent)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        exponent_shape_{exponent.shape()},
        exponent_strides_{exponent.strides()},
        exponent_type_{exponent.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor exponent,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const double exponent,
                          Tensor out) const = 0;

  /// \deprecated Use an explicit-output `operator()` overload. This overload
  /// will be removed in a future release.
  [[deprecated("Use an explicit-output overload instead.")]]
  virtual void operator()(Tensor input, const double exponent) const = 0;

  /// \deprecated Use an explicit-output `operator()` overload. This overload
  /// will be removed in a future release.
  [[deprecated("Use an explicit-output overload instead.")]]
  virtual void operator()(Tensor input, const Tensor exponent) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape exponent_shape_;

  Tensor::Strides exponent_strides_;

  DataType exponent_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double exponent_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
