#ifndef INFINI_OPS_BASE_SPECIAL_ROUND_H_
#define INFINI_OPS_BASE_SPECIAL_ROUND_H_

#include "operator.h"

namespace infini::ops::special {

class Round : public Operator<Round> {
 public:
  Round(const Tensor input, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        decimals_{0},
        device_index_{out.device().index()} {}

  /// \deprecated Use `(input, out)`. This constructor will be removed in a
  /// future release.
  [[deprecated("Use the `(input, out)` overload instead.")]]
  Round(const Tensor input, const int64_t decimals, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        decimals_{decimals},
        device_index_{out.device().index()} {}

  void operator()(const Tensor input, Tensor out) const {
    (*this)(input, 0, out);
  }

  /// \deprecated Use `operator()(input, out)`. This overload will be removed
  /// in a future release.
  [[deprecated("Use the `(input, out)` overload instead.")]]
  virtual void operator()(const Tensor input, const int64_t decimals,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t decimals_{};

  int device_index_{0};
};

}  // namespace infini::ops::special

#endif
