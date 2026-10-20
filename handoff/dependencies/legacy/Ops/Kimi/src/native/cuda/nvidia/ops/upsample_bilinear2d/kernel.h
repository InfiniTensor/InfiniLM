#ifndef INFINI_OPS_NVIDIA_UPSAMPLE_BILINEAR2D_KERNEL_H_
#define INFINI_OPS_NVIDIA_UPSAMPLE_BILINEAR2D_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "base/upsample_bilinear2d.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {
namespace nvidia_upsample_bilinear2d_detail {
struct Layout {
  std::size_t channels;
  std::size_t input_height;
  std::size_t input_width;
  std::size_t output_height;
  std::size_t output_width;
  std::size_t input_strides[4];
  float scale_height;
  float scale_width;
  bool align_corners;
};
void Launch(const void* input, void* out, DataType dtype, Layout layout,
            std::size_t output_count, cudaStream_t stream);
}  // namespace nvidia_upsample_bilinear2d_detail

template <>
class Operator<UpsampleBilinear2d, Device::Type::kNvidia, 0>
    : public UpsampleBilinear2d {
 public:
  static bool supports(const Tensor input,
                       const std::vector<std::int64_t> output_size,
                       const bool align_corners,
                       const std::optional<double> scales_h,
                       const std::optional<double> scales_w, Tensor out) {
    (void)align_corners;
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != out.device() ||
        input.shape().size() != 4 || out.shape().size() != 4 ||
        input.strides().size() != 4 || out.strides().size() != 4 ||
        output_size.size() != 2 || output_size[0] <= 0 || output_size[1] <= 0 ||
        scales_h || scales_w || input.dtype() != out.dtype() ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32) ||
        input.shape()[0] != out.shape()[0] ||
        input.shape()[1] != out.shape()[1]) return false;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != input.device().index()) return false;
    if (input.dtype() == DataType::kBFloat16) {
      cudaDeviceProp properties{};
      if (cudaGetDeviceProperties(&properties, current_device) != cudaSuccess ||
          properties.major < 8) return false;
    }
    const auto max_spatial = static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max());
    if (static_cast<std::uint64_t>(output_size[0]) > max_spatial ||
        static_cast<std::uint64_t>(output_size[1]) > max_spatial ||
        out.shape()[2] != static_cast<std::size_t>(output_size[0]) ||
        out.shape()[3] != static_cast<std::size_t>(output_size[1]) ||
        input.shape()[2] > max_spatial || input.shape()[3] > max_spatial)
      return false;

    std::size_t input_span = 1;
    std::size_t output_count = 1;
    for (std::size_t d = 0; d < 4; ++d) {
      const auto extent = input.shape()[d];
      const auto stride = input.strides()[d];
      if (!extent || !out.shape()[d] || stride < 0 ||
          (extent > 1 && stride == 0) ||
          static_cast<std::uint64_t>(stride) >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
          extent > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
          out.shape()[d] > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
          (extent - 1 > (std::numeric_limits<std::size_t>::max() - input_span) /
                            static_cast<std::size_t>(stride ? stride : 1)) ||
          output_count > std::numeric_limits<std::size_t>::max() / out.shape()[d])
        return false;
      input_span += (extent - 1) * static_cast<std::size_t>(stride);
      output_count *= out.shape()[d];
    }
    const auto width = input.element_size();
    if (!width || input_span > std::numeric_limits<std::size_t>::max() / width ||
        output_count > std::numeric_limits<std::size_t>::max() / width ||
        !CheckedContiguous(out) || !input.data() || !out.data() ||
        reinterpret_cast<std::uintptr_t>(input.data()) % width ||
        reinterpret_cast<std::uintptr_t>(out.data()) % width)
      return false;
    return !Overlaps(input.data(), input_span * width, out.data(),
                     output_count * width);
  }

  Operator(const Tensor input, const std::vector<std::int64_t> output_size,
           const bool align_corners, const std::optional<double> scales_h,
           const std::optional<double> scales_w, Tensor out)
      : UpsampleBilinear2d(input, output_size, align_corners, scales_h,
                           scales_w, out) {
    Validate(input, output_size, align_corners, scales_h, scales_w, out);
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input,
                  const std::vector<std::int64_t> output_size,
                  const bool align_corners,
                  const std::optional<double> scales_h,
                  const std::optional<double> scales_w, Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        input.dtype() != input_type_ || out.shape() != out_shape_ ||
        out.strides() != out_strides_ || out.dtype() != out_type_ ||
        output_size != output_size_ || align_corners != align_corners_ ||
        scales_h != scales_h_ || scales_w != scales_w_ ||
        out.device().index() != device_index_)
      throw std::invalid_argument(
          "NVIDIA UpsampleBilinear2d metadata changed after Make");
    Validate(input, output_size, align_corners, scales_h, scales_w, out);
    nvidia_upsample_bilinear2d_detail::Layout layout{};
    layout.channels = input.shape()[1];
    layout.input_height = input.shape()[2];
    layout.input_width = input.shape()[3];
    layout.output_height = out.shape()[2];
    layout.output_width = out.shape()[3];
    for (std::size_t d = 0; d < 4; ++d)
      layout.input_strides[d] = static_cast<std::size_t>(input.strides()[d]);
    layout.align_corners = align_corners;
    layout.scale_height = align_corners
        ? (layout.output_height > 1
               ? static_cast<float>(layout.input_height - 1) /
                     static_cast<float>(layout.output_height - 1)
               : 0.0F)
        : static_cast<float>(layout.input_height) /
              static_cast<float>(layout.output_height);
    layout.scale_width = align_corners
        ? (layout.output_width > 1
               ? static_cast<float>(layout.input_width - 1) /
                     static_cast<float>(layout.output_width - 1)
               : 0.0F)
        : static_cast<float>(layout.input_width) /
              static_cast<float>(layout.output_width);
    nvidia_upsample_bilinear2d_detail::Launch(
        input.data(), out.data(), input.dtype(), layout, out.numel(),
        static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  static bool CheckedContiguous(const Tensor tensor) {
    std::size_t expected = 1;
    for (std::size_t i = 4; i > 0; --i) {
      if (expected > static_cast<std::size_t>(
                         std::numeric_limits<Tensor::Stride>::max()) ||
          tensor.strides()[i - 1] != static_cast<Tensor::Stride>(expected) ||
          expected > std::numeric_limits<std::size_t>::max() /
                         tensor.shape()[i - 1]) return false;
      expected *= tensor.shape()[i - 1];
    }
    return true;
  }
  static bool Overlaps(const void* a, std::size_t a_bytes,
                       const void* b, std::size_t b_bytes) {
    const auto first = reinterpret_cast<std::uintptr_t>(a);
    const auto second = reinterpret_cast<std::uintptr_t>(b);
    const auto max = std::numeric_limits<std::uintptr_t>::max();
    return a_bytes > max - first || b_bytes > max - second ||
           (first < second + b_bytes && second < first + a_bytes);
  }
  static void Validate(const Tensor input,
                       const std::vector<std::int64_t>& output_size,
                       bool align_corners, std::optional<double> scales_h,
                       std::optional<double> scales_w, Tensor out) {
    if (!supports(input, output_size, align_corners, scales_h, scales_w, out))
      throw std::invalid_argument(
          "unsupported NVIDIA UpsampleBilinear2d metadata, device or alias");
  }
};
}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_UPSAMPLE_BILINEAR2D_KERNEL_H_
