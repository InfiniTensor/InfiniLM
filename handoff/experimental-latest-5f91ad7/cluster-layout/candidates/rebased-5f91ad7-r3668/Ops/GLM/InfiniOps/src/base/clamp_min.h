#ifndef INFINI_OPS_BASE_CLAMP_MIN_H_
#define INFINI_OPS_BASE_CLAMP_MIN_H_

#include "operator.h"

namespace infini::ops {

class ClampMin : public Operator<ClampMin> {
 public:
  ClampMin(const Tensor input, const double min, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        min_{min},
        device_index_{out.device().index()} {}

  ClampMin(const Tensor input, const Tensor min, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        min_shape_{min.shape()},
        min_strides_{min.strides()},
        min_type_{min.dtype()},
        device_index_{out.device().index()} {}

  /// \deprecated Use the explicit-output constructor. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the explicit-output overload instead.")]]
  ClampMin(Tensor input, const Tensor min)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        min_shape_{min.shape()},
        min_strides_{min.strides()},
        min_type_{min.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(const Tensor input, const double min,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor min,
                          Tensor out) const = 0;

  /// \deprecated Use an explicit-output `operator()` overload. This overload
  /// will be removed in a future release.
  [[deprecated("Use an explicit-output overload instead.")]]
  virtual void operator()(Tensor input, const Tensor min) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double min_{};

  Tensor::Shape min_shape_;

  Tensor::Strides min_strides_;

  DataType min_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
