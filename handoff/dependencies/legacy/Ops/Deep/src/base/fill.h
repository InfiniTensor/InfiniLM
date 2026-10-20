#ifndef INFINI_OPS_BASE_FILL_H_
#define INFINI_OPS_BASE_FILL_H_


#include "data_type.h"
#include "operator.h"

namespace infini::ops {

class Fill : public Operator<Fill> {
 public:
  // Output metadata contract; require storage at Call/RT memset, not planning.
  // This does not select a provider or qualify its scalar/capture domain.
  static bool valid_output(const Tensor& out, bool require_storage = true) {
    const auto dtype = static_cast<int>(out.dtype());
    if (dtype < 0 || dtype > static_cast<int>(DataType::kFloat64) ||
        out.shape().size() != out.strides().size()) return false;
    for (std::size_t i = 0; i < out.ndim(); ++i)
      if (out.size(i) > 1 && out.stride(i) == 0) return false;
    return !require_storage || !out.numel() || out.data();
  }

  static bool supports(const Tensor& input, double, const Tensor& out) {
    return input.shape().size() == input.strides().size() &&
           input.shape() == out.shape() && input.dtype() == out.dtype() &&
           input.device() == out.device() && valid_output(out, false);
  }

  Fill(const Tensor input, const double value, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        value_{value},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  Fill(const Tensor input, const Tensor value, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        value_shape_{value.shape()},
        value_strides_{value.strides()},
        value_type_{value.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const double value,
                          Tensor out) const = 0;

  virtual void operator()(const Tensor input, const Tensor value,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  double value_{};

  Tensor::Shape value_shape_;

  Tensor::Strides value_strides_;

  DataType value_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int device_index_{0};

};

}  // namespace infini::ops

#endif
