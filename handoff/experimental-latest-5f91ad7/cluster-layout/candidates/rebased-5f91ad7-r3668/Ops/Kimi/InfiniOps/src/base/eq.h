#ifndef INFINI_OPS_BASE_EQ_H_
#define INFINI_OPS_BASE_EQ_H_

#include "operator.h"

namespace infini::ops {

class Eq : public Operator<Eq> {
 public:
  Eq(const Tensor input, const double other, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        other_{other},
        device_index_{out.device().index()} {}

  Eq(const Tensor input, const Tensor other, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        device_index_{out.device().index()} {}

  /// \deprecated Use the explicit-output constructor. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the explicit-output overload instead.")]]
  Eq(Tensor input, const double other)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_{other},
        device_index_{input.device().index()} {}

  /// \deprecated Use the explicit-output constructor. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the explicit-output overload instead.")]]
  Eq(Tensor input, const Tensor other)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        other_shape_{other.shape()},
        other_strides_{other.strides()},
        other_type_{other.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(const Tensor input, const double other,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor other,
                          Tensor out) const = 0;

  /// \deprecated Use an explicit-output `operator()` overload. This overload
  /// will be removed in a future release.
  [[deprecated("Use an explicit-output overload instead.")]]
  virtual void operator()(Tensor input, const double other) const = 0;

  /// \deprecated Use an explicit-output `operator()` overload. This overload
  /// will be removed in a future release.
  [[deprecated("Use an explicit-output overload instead.")]]
  virtual void operator()(Tensor input, const Tensor other) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double other_{};

  Tensor::Shape other_shape_;

  Tensor::Strides other_strides_;

  DataType other_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
