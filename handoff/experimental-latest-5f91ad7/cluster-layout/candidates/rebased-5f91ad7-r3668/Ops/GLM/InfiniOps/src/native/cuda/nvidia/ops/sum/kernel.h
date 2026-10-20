#ifndef INFINI_OPS_NVIDIA_SUM_KERNEL_H_
#define INFINI_OPS_NVIDIA_SUM_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "base/sum.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {
namespace nvidia_sum_detail {
struct Layout {
  std::int64_t shape[4]{};
  std::int64_t stride[4]{};
  std::int32_t rank{};
  std::int32_t axis{};
  std::int64_t reduce_length{};
};
void Launch(const void* input, void* out, DataType dtype, Layout layout,
            std::size_t output_count, cudaStream_t stream);
}  // namespace nvidia_sum_detail

template <>
class Operator<Sum, Device::Type::kNvidia, 0> : public Sum {
 public:
  static bool supports(const Tensor input,
                       const std::optional<std::vector<std::int64_t>> dim,
                       const bool keepdim, const std::optional<DataType> dtype,
                       Tensor out) {
    return Supports(input, dim, keepdim, dtype, out, false);
  }

 private:
  static bool Supports(const Tensor input,
                       const std::optional<std::vector<std::int64_t>> dim,
                       const bool keepdim, const std::optional<DataType> dtype,
                       Tensor out, bool invocation) {
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        !dim || dim->size() != 1 || keepdim ||
        (dtype && *dtype != input.dtype()) || input.shape().size() < 1 ||
        input.shape().size() > 4 ||
        out.shape().size() + 1 != input.shape().size())
      return false;
    if (input.dtype() != DataType::kFloat16 &&
        input.dtype() != DataType::kBFloat16 &&
        input.dtype() != DataType::kFloat32)
      return false;
    if (input.dtype() == DataType::kBFloat16) {
      cudaDeviceProp properties{};
      if (cudaGetDeviceProperties(&properties, input.device().index()) !=
              cudaSuccess ||
          properties.major < 8)
        return false;
    }
    const auto rank = input.shape().size();
    if ((*dim)[0] < 0 || static_cast<std::size_t>((*dim)[0]) >= rank)
      return false;
    const auto axis = static_cast<std::size_t>((*dim)[0]);
    if (input.shape()[axis] == 0 || input.strides().size() != rank ||
        out.strides().size() != out.shape().size())
      return false;
    std::size_t output_count = 1;
    std::size_t output_dimension = 0;
    std::size_t last_element = 0;
    for (std::size_t d = 0; d < rank; ++d) {
      const auto extent = input.shape()[d];
      const auto stride = input.strides()[d];
      if (!extent || stride < 0 ||
          extent > static_cast<std::size_t>(
                       std::numeric_limits<std::int64_t>::max()) ||
          static_cast<std::uint64_t>(stride) >
              static_cast<std::uint64_t>(
                  std::numeric_limits<std::int64_t>::max()))
        return false;
      if (extent - 1 >
          (std::numeric_limits<std::size_t>::max() - last_element) /
              static_cast<std::size_t>(stride ? stride : 1))
        return false;
      last_element += (extent - 1) * static_cast<std::size_t>(stride);
      if (d != axis) {
        if (out.shape()[output_dimension++] != extent ||
            output_count > std::numeric_limits<std::size_t>::max() / extent)
          return false;
        output_count *= extent;
      }
    }
    const auto width = input.element_size();
    if (!width ||
        last_element > static_cast<std::size_t>(
                           std::numeric_limits<std::int64_t>::max()) ||
        last_element == std::numeric_limits<std::size_t>::max() ||
        last_element + 1 > std::numeric_limits<std::size_t>::max() / width ||
        output_count > std::numeric_limits<std::int64_t>::max() ||
        output_count > std::numeric_limits<std::size_t>::max() / width ||
        !CheckedContiguous(out))
      return false;
    if (!invocation) return true;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != input.device().index() || !input.data() ||
        !out.data() || reinterpret_cast<std::uintptr_t>(input.data()) % width ||
        reinterpret_cast<std::uintptr_t>(out.data()) % width)
      return false;
    return !Overlaps(input.data(), (last_element + 1) * width, out.data(),
                     output_count * width);
  }

 public:
  Operator(const Tensor input,
           const std::optional<std::vector<std::int64_t>> dim,
           const bool keepdim, const std::optional<DataType> dtype, Tensor out)
      : Sum(input, dim, keepdim, dtype, out) {
    Validate(input, dim, keepdim, dtype, out);
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input,
                  const std::optional<std::vector<std::int64_t>> dim,
                  const bool keepdim, const std::optional<DataType> dtype,
                  Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        input.dtype() != input_type_ || out.shape() != out_shape_ ||
        out.strides() != out_strides_ || out.dtype() != out_type_ ||
        dim != dim_ || keepdim != keepdim_ || dtype != dtype_ ||
        out.device().index() != device_index_) {
      throw std::invalid_argument("NVIDIA Sum metadata changed after Make");
    }
    Validate(input, dim, keepdim, dtype, out);
    nvidia_sum_detail::Layout layout;
    layout.rank = static_cast<std::int32_t>(input.shape().size());
    layout.axis = static_cast<std::int32_t>((*dim)[0]);
    layout.reduce_length =
        static_cast<std::int64_t>(input.shape()[layout.axis]);
    for (std::int32_t d = 0; d < layout.rank; ++d) {
      layout.shape[d] = static_cast<std::int64_t>(input.shape()[d]);
      layout.stride[d] = static_cast<std::int64_t>(input.strides()[d]);
    }
    nvidia_sum_detail::Launch(input.data(), out.data(), input.dtype(), layout,
                              out.numel(),
                              static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  static bool CheckedContiguous(const Tensor tensor) {
    std::size_t expected = 1;
    for (std::size_t i = tensor.shape().size(); i > 0; --i) {
      if (tensor.shape()[i - 1] == 1) continue;
      if (expected > static_cast<std::size_t>(
                         std::numeric_limits<Tensor::Stride>::max()) ||
          tensor.strides()[i - 1] != static_cast<Tensor::Stride>(expected))
        return false;
      if (tensor.shape()[i - 1] &&
          expected >
              std::numeric_limits<std::size_t>::max() / tensor.shape()[i - 1])
        return false;
      expected *= tensor.shape()[i - 1];
    }
    return true;
  }
  static bool Overlaps(const void* a, std::size_t a_bytes, const void* b,
                       std::size_t b_bytes) {
    const auto first = reinterpret_cast<std::uintptr_t>(a);
    const auto second = reinterpret_cast<std::uintptr_t>(b);
    const auto max = std::numeric_limits<std::uintptr_t>::max();
    return a_bytes > max - first || b_bytes > max - second ||
           (first < second + b_bytes && second < first + a_bytes);
  }
  static void Validate(const Tensor input,
                       const std::optional<std::vector<std::int64_t>> dim,
                       const bool keepdim, const std::optional<DataType> dtype,
                       Tensor out) {
    if (!Supports(input, dim, keepdim, dtype, out, true)) {
      throw std::invalid_argument(
          "unsupported NVIDIA Sum metadata, device or alias");
    }
  }
};
}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_SUM_KERNEL_H_
