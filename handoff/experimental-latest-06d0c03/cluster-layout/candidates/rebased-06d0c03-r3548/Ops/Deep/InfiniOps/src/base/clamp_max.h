#ifndef INFINI_OPS_BASE_CLAMP_MAX_H_
#define INFINI_OPS_BASE_CLAMP_MAX_H_

#include "operator.h"

namespace infini::ops {

class ClampMax : public Operator<ClampMax> {
 public:
  ClampMax(const Tensor input, const double max, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        max_{max},
        device_index_{out.device().index()} {}

  ClampMax(const Tensor input, const Tensor max, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        device_index_{out.device().index()} {}

  /// \deprecated Use the explicit-output constructor. This constructor will be
  /// removed in a future release.
  [[deprecated("Use the explicit-output overload instead.")]]
  ClampMax(Tensor input, const Tensor max)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        max_shape_{max.shape()},
        max_strides_{max.strides()},
        max_type_{max.dtype()},
        device_index_{input.device().index()} {}

  virtual void operator()(const Tensor input, const double max,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor max,
                          Tensor out) const = 0;

  /// \deprecated Use an explicit-output `operator()` overload. This overload
  /// will be removed in a future release.
  [[deprecated("Use an explicit-output overload instead.")]]
  virtual void operator()(Tensor input, const Tensor max) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double max_{};

  Tensor::Shape max_shape_;

  Tensor::Strides max_strides_;

  DataType max_type_;

  int device_index_{0};
};

}  // namespace infini::ops

#endif
