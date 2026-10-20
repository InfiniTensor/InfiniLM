#ifndef INFINI_OPS_BASE_LINEAR_H_
#define INFINI_OPS_BASE_LINEAR_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

class Linear : public Operator<Linear> {
 public:
  Linear(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
         Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        rows_{Rows(input)},
        has_bias_{bias.has_value()} {
    assert(input.ndim() >= 1 && "operator `Linear` requires non-scalar input");
    assert(weight.ndim() == 2 && "operator `Linear` requires 2D `weight`");
    assert(input.size(-1) == weight.size(-1) &&
           "operator `Linear` input features must match `weight`");
    assert(out.ndim() == input.ndim() &&
           "operator `Linear` output rank must match input rank");
    assert(out.size(-1) == weight.size(0) &&
           "operator `Linear` output features must match `weight`");

    for (Tensor::Size axis = 0; axis + 1 < input.ndim(); ++axis) {
      assert(input.size(axis) == out.size(axis) &&
             "operator `Linear` output leading dimensions must match input");
    }

    assert(input.dtype() == weight.dtype() &&
           "operator `Linear` requires input and weight to have the same "
           "dtype");
    assert(input.dtype() == out.dtype() &&
           "operator `Linear` requires output to have the input dtype");
    if (has_bias_) {
      assert(bias->ndim() == 1 && bias->size(0) == weight.size(0) &&
             "operator `Linear` bias must have shape `[out_features]`");
      assert(bias->dtype() == out.dtype() &&
             "operator `Linear` requires bias to have the output dtype");
    }
  }

  virtual void operator()(const Tensor input, const Tensor weight,
                          std::optional<Tensor> bias, Tensor out) const = 0;

 protected:
  static Tensor::Size Rows(const Tensor input) {
    Tensor::Size rows = 1;

    for (Tensor::Size axis = 0; axis + 1 < input.ndim(); ++axis) {
      rows *= input.size(axis);
    }

    return input.ndim() == 0 ? 0 : rows;
  }

  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  Tensor::Size rows_{0};

  bool has_bias_{false};
};

}  // namespace infini::ops

#endif
