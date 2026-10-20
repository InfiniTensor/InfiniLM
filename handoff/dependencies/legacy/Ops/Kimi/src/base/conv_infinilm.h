#ifndef INFINI_OPS_BASE_CONV_INFINILM_H_
#define INFINI_OPS_BASE_CONV_INFINILM_H_

#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

/// \deprecated Migrate to an open-source-aligned operator when available. This
/// interface will be removed in a future release.
class [[deprecated(
    "Migrate to an open-source-aligned operator when available.")]] ConvInfinilm
    : public Operator<ConvInfinilm> {
 public:
  ConvInfinilm(const Tensor input, const Tensor weight,
               std::optional<Tensor> bias, const std::vector<int64_t> padding,
               const std::vector<int64_t> stride,
               const std::vector<int64_t> dilation, const int64_t groups,
               Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        bias_shape_{bias.has_value() ? Tensor::Shape{bias->shape()}
                                     : Tensor::Shape{}},
        bias_strides_{bias.has_value() ? Tensor::Strides{bias->strides()}
                                       : Tensor::Strides{}},
        input_type_{input.dtype()},
        weight_type_{weight.dtype()},
        out_type_{out.dtype()},
        bias_type_{bias.has_value() ? bias->dtype() : out.dtype()},
        padding_{padding},
        stride_{stride},
        dilation_{dilation},
        groups_{groups},
        spatial_ndim_{input.ndim() - 2},
        output_size_{out.numel()},
        kernel_size_{1},
        device_index_{out.device().index()},
        has_bias_{bias.has_value()} {
    assert(input.ndim() >= 3 && input.ndim() <= 5 &&
           "`ConvInfinilm` supports 1D, 2D, and 3D conv_infinilmolution");
    assert(input.ndim() == weight.ndim() && input.ndim() == out.ndim() &&
           "`ConvInfinilm` input, weight, and output ranks must match");
    assert(padding.size() == spatial_ndim_ && stride.size() == spatial_ndim_ &&
           dilation.size() == spatial_ndim_ &&
           "`ConvInfinilm` padding, stride, and dilation rank mismatch");
    assert(groups > 0 && "`ConvInfinilm` groups must be positive");
    assert(input_type_ == weight_type_ && input_type_ == out_type_ &&
           "`ConvInfinilm` input, weight, and output dtypes must match");
    assert(input_shape_[1] % groups == 0 &&
           "`ConvInfinilm` input channels must be divisible by groups");
    assert(weight_shape_[0] % groups == 0 &&
           "`ConvInfinilm` output channels must be divisible by groups");
    assert(weight_shape_[1] == input_shape_[1] / groups &&
           "`ConvInfinilm` weight input channels mismatch");
    assert(out_shape_[0] == input_shape_[0] &&
           "`ConvInfinilm` output batch size mismatch");
    assert(out_shape_[1] == weight_shape_[0] &&
           "`ConvInfinilm` output channels mismatch");
    assert(!out.HasBroadcastDim() &&
           "`ConvInfinilm` output must not have broadcasted dimensions");

    if (has_bias_) {
      assert(bias_type_ == out_type_ && "`ConvInfinilm` bias dtype mismatch");
      assert(bias_shape_.size() == 1 && bias_shape_[0] == out_shape_[1] &&
             "`ConvInfinilm` bias shape must be `(out_channels,)`");
    }

    for (std::size_t i = 0; i < spatial_ndim_; ++i) {
      assert(stride_[i] > 0 && "`ConvInfinilm` stride values must be positive");
      assert(dilation_[i] > 0 &&
             "`ConvInfinilm` dilation values must be positive");
      assert(padding_[i] >= 0 &&
             "`ConvInfinilm` padding values must be non-negative");

      const auto expected = (input_shape_[i + 2] + 2 * padding_[i] -
                             dilation_[i] * (weight_shape_[i + 2] - 1) - 1) /
                                stride_[i] +
                            1;
      assert(out_shape_[i + 2] == expected &&
             "`ConvInfinilm` output spatial shape mismatch");
      kernel_size_ *= weight_shape_[i + 2];
    }
  }

  virtual void operator()(const Tensor input, const Tensor weight,
                          std::optional<Tensor> bias,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> dilation,
                          const int64_t groups, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  Tensor::Shape bias_shape_;

  Tensor::Strides bias_strides_;

  DataType input_type_;

  DataType weight_type_;

  DataType out_type_;

  DataType bias_type_;

  std::vector<int64_t> padding_;

  std::vector<int64_t> stride_;

  std::vector<int64_t> dilation_;

  int64_t groups_{1};

  Tensor::Size spatial_ndim_{0};

  Tensor::Size output_size_{0};

  Tensor::Size kernel_size_{1};

  int device_index_{0};

  bool has_bias_{false};
};

}  // namespace infini::ops

#endif
