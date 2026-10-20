#ifndef INFINI_OPS_BASE_SLOW_CONV3D_FORWARD_H_
#define INFINI_OPS_BASE_SLOW_CONV3D_FORWARD_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

class SlowConv3dForward : public Operator<SlowConv3dForward> {
 public:
  SlowConv3dForward(const Tensor input, const Tensor weight,
                    const std::optional<Tensor> bias,
                    const std::vector<int64_t> kernel_size,
                    const std::vector<int64_t> stride,
                    const std::vector<int64_t> padding, Tensor output)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        output_shape_{output.shape()},
        output_strides_{output.strides()},
        output_type_{output.dtype()},
        has_bias_{bias.has_value()},
        bias_shape_{bias ? Tensor::Shape{bias->shape()} : Tensor::Shape{}},
        bias_strides_{bias ? Tensor::Strides{bias->strides()}
                           : Tensor::Strides{}},
        bias_type_{bias ? bias->dtype() : DataType::kFloat32},
        kernel_size_{kernel_size},
        stride_{stride},
        padding_{padding},
        device_index_{output.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor weight,
                          const std::optional<Tensor> bias,
                          const std::vector<int64_t> kernel_size,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          Tensor output) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  Tensor::Shape output_shape_;

  Tensor::Strides output_strides_;

  DataType output_type_;

  bool has_bias_{false};

  Tensor::Shape bias_shape_;

  Tensor::Strides bias_strides_;

  DataType bias_type_{DataType::kFloat32};

  std::vector<int64_t> kernel_size_{};

  std::vector<int64_t> stride_{};

  std::vector<int64_t> padding_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
