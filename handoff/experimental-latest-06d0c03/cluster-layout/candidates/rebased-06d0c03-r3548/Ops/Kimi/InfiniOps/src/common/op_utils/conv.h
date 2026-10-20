#ifndef INFINI_OPS_COMMON_OP_UTILS_CONV_H_
#define INFINI_OPS_COMMON_OP_UTILS_CONV_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "operator.h"

namespace infini::ops::conv_detail {

struct Padding {
  std::vector<int64_t> left;
  std::vector<int64_t> total;
};

struct Metadata {
  Tensor::Shape input_shape;
  Tensor::Strides input_strides;
  Tensor::Shape weight_shape;
  Tensor::Strides weight_strides;
  Tensor::Shape out_shape;
  Tensor::Strides out_strides;
  Tensor::Shape bias_shape;
  Tensor::Strides bias_strides;
  DataType input_type;
  DataType weight_type;
  DataType out_type;
  DataType bias_type;
  std::vector<int64_t> padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups{1};
  Tensor::Size spatial_ndim{0};
  Tensor::Size output_size{0};
  Tensor::Size kernel_size{1};
  int device_index{0};
  bool has_bias{false};
};

template <std::size_t SpatialDimensions>
Padding ResolvePadding(const Tensor weight, const std::vector<int64_t>& stride,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& dilation) {
  assert(weight.ndim() == SpatialDimensions + 2 &&
         "operator `Conv` weight rank must match its spatial dimensions");
  assert(stride.size() == SpatialDimensions &&
         "operator `Conv` `stride` has the wrong length");
  assert(padding.size() == SpatialDimensions &&
         "operator `Conv` `padding` has the wrong length");
  assert(dilation.size() == SpatialDimensions &&
         "operator `Conv` `dilation` has the wrong length");

  Padding resolved{padding, std::vector<int64_t>(SpatialDimensions)};

  for (std::size_t dim = 0; dim < SpatialDimensions; ++dim) {
    assert(stride[dim] > 0 &&
           "operator `Conv` requires positive `stride` values");
    assert(padding[dim] >= 0 &&
           "operator `Conv` requires non-negative `padding` values");
    assert(dilation[dim] > 0 &&
           "operator `Conv` requires positive `dilation` values");
    resolved.total[dim] = 2 * padding[dim];
  }

  return resolved;
}

template <std::size_t SpatialDimensions>
Padding ResolvePadding(const Tensor weight, const std::vector<int64_t>& stride,
                       const std::string& padding,
                       const std::vector<int64_t>& dilation) {
  assert(weight.ndim() == SpatialDimensions + 2 &&
         "operator `Conv` weight rank must match its spatial dimensions");
  assert(stride.size() == SpatialDimensions &&
         "operator `Conv` `stride` has the wrong length");
  assert(dilation.size() == SpatialDimensions &&
         "operator `Conv` `dilation` has the wrong length");
  assert((padding == "valid" || padding == "same") &&
         "operator `Conv` string `padding` must be `valid` or `same`");

  Padding resolved{std::vector<int64_t>(SpatialDimensions),
                   std::vector<int64_t>(SpatialDimensions)};

  for (std::size_t dim = 0; dim < SpatialDimensions; ++dim) {
    assert(stride[dim] > 0 &&
           "operator `Conv` requires positive `stride` values");
    assert(dilation[dim] > 0 &&
           "operator `Conv` requires positive `dilation` values");

    if (padding == "same") {
      assert(stride[dim] == 1 &&
             "operator `Conv` does not support `same` padding with strides");
      resolved.total[dim] =
          dilation[dim] * (static_cast<int64_t>(weight.size(dim + 2)) - 1);
      resolved.left[dim] = resolved.total[dim] / 2;
    }
  }

  return resolved;
}

template <std::size_t SpatialDimensions>
Metadata MakeMetadata(const Tensor input, const Tensor weight,
                      std::optional<Tensor> bias,
                      const std::vector<int64_t>& stride,
                      const Padding& padding,
                      const std::vector<int64_t>& dilation,
                      const int64_t groups, Tensor out) {
  Metadata metadata;
  metadata.input_shape = Tensor::Shape{input.shape()};
  metadata.input_strides = Tensor::Strides{input.strides()};
  metadata.weight_shape = Tensor::Shape{weight.shape()};
  metadata.weight_strides = Tensor::Strides{weight.strides()};
  metadata.out_shape = Tensor::Shape{out.shape()};
  metadata.out_strides = Tensor::Strides{out.strides()};
  metadata.bias_shape =
      bias.has_value() ? Tensor::Shape{bias->shape()} : Tensor::Shape{};
  metadata.bias_strides =
      bias.has_value() ? Tensor::Strides{bias->strides()} : Tensor::Strides{};
  metadata.input_type = input.dtype();
  metadata.weight_type = weight.dtype();
  metadata.out_type = out.dtype();
  metadata.bias_type = bias.has_value() ? bias->dtype() : out.dtype();
  metadata.padding = padding.left;
  metadata.stride = stride;
  metadata.dilation = dilation;
  metadata.groups = groups;
  metadata.spatial_ndim = SpatialDimensions;
  metadata.output_size = out.numel();
  metadata.device_index = out.device().index();
  metadata.has_bias = bias.has_value();

  assert(input.ndim() == SpatialDimensions + 2 &&
         "operator `Conv` input rank must match its spatial dimensions");
  assert(weight.ndim() == input.ndim() && out.ndim() == input.ndim() &&
         "operator `Conv` input, weight, and output ranks must match");
  assert(padding.left.size() == SpatialDimensions &&
         padding.total.size() == SpatialDimensions &&
         "operator `Conv` resolved padding has the wrong length");
  assert(groups > 0 && "operator `Conv` requires positive `groups`");
  assert(metadata.input_type == metadata.weight_type &&
         metadata.input_type == metadata.out_type &&
         "operator `Conv` input, weight, and output dtypes must match");
  assert(input.device() == weight.device() && input.device() == out.device() &&
         "operator `Conv` input, weight, and output devices must match");
  assert(metadata.input_shape[1] % groups == 0 &&
         "operator `Conv` input channels must be divisible by `groups`");
  assert(metadata.weight_shape[0] % groups == 0 &&
         "operator `Conv` output channels must be divisible by `groups`");
  assert(metadata.weight_shape[1] == metadata.input_shape[1] / groups &&
         "operator `Conv` weight input channels do not match `groups`");
  assert(metadata.out_shape[0] == metadata.input_shape[0] &&
         "operator `Conv` output batch size must match input");
  assert(metadata.out_shape[1] == metadata.weight_shape[0] &&
         "operator `Conv` output channels must match weight");
  assert(!out.HasBroadcastDim() &&
         "operator `Conv` output must not have broadcasted dimensions");

  if (metadata.has_bias) {
    assert(bias->device() == out.device() &&
           "operator `Conv` bias and output devices must match");
    assert(metadata.bias_type == metadata.out_type &&
           "operator `Conv` bias and output dtypes must match");
    assert(metadata.bias_shape.size() == 1 &&
           metadata.bias_shape[0] == metadata.out_shape[1] &&
           "operator `Conv` bias must have shape `(out_channels,)`");
  }

  for (std::size_t dim = 0; dim < SpatialDimensions; ++dim) {
    const auto input_size = static_cast<int64_t>(metadata.input_shape[dim + 2]);
    const auto kernel_size =
        static_cast<int64_t>(metadata.weight_shape[dim + 2]);
    const auto numerator =
        input_size + padding.total[dim] - dilation[dim] * (kernel_size - 1) - 1;
    assert(numerator >= 0 &&
           "operator `Conv` kernel exceeds the padded input size");
    const auto expected = numerator / stride[dim] + 1;
    assert(metadata.out_shape[dim + 2] == static_cast<Tensor::Size>(expected) &&
           "operator `Conv` output spatial shape is incorrect");
    metadata.kernel_size *= metadata.weight_shape[dim + 2];
  }

  return metadata;
}

}  // namespace infini::ops::conv_detail

#endif
