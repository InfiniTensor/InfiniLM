#ifndef INFINI_OPS_BASE_LAYER_NORM_H_
#define INFINI_OPS_BASE_LAYER_NORM_H_

#include <optional>

#include "operator.h"

namespace infini::ops {

// Inference LayerNorm over the last dimension. The legacy Core training
// intermediates are deliberately not part of this operator's outputs.
class LayerNorm : public Operator<LayerNorm> {
 public:
  static constexpr bool kAutomaticDefault = true;

  LayerNorm(const Tensor input, const Tensor weight,
            const std::optional<Tensor> bias, float eps, Tensor out)
      : input_shape_{input.shape()}, input_strides_{input.strides()},
        weight_shape_{weight.shape()}, weight_strides_{weight.strides()},
        bias_shape_{bias ? Tensor::Shape{bias->shape()} : Tensor::Shape{}},
        bias_strides_{bias ? Tensor::Strides{bias->strides()} : Tensor::Strides{}},
        out_shape_{out.shape()}, out_strides_{out.strides()},
        dtype_{input.dtype()}, device_{input.device()}, has_bias_{bias.has_value()},
        eps_{eps} {}

  virtual void operator()(const Tensor input, const Tensor weight,
                          const std::optional<Tensor> bias, float eps,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_, weight_shape_, bias_shape_, out_shape_;
  Tensor::Strides input_strides_, weight_strides_, bias_strides_, out_strides_;
  DataType dtype_;
  Device device_;
  bool has_bias_;
  float eps_;
};

}  // namespace infini::ops

#endif
