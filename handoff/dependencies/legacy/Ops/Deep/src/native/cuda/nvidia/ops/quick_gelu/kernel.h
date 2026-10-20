#ifndef INFINI_OPS_NVIDIA_QUICK_GELU_KERNEL_H_
#define INFINI_OPS_NVIDIA_QUICK_GELU_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "base/quick_gelu.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {
namespace nvidia_quick_gelu_detail {
struct Layout {
  std::size_t shape[8];
  std::size_t input_strides[8];
  std::size_t out_strides[8];
  unsigned rank;
};
void Launch(const void* input, void* out, DataType dtype, Layout layout,
            std::size_t count, cudaStream_t stream);
}  // namespace nvidia_quick_gelu_detail

template <>
class Operator<QuickGelu, Device::Type::kNvidia, 0> : public QuickGelu {
 public:
  static bool supports(const Tensor input, Tensor out) {
    using namespace nvidia_binary_view;
    const auto rank = input.shape().size();
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != out.device() || rank < 1 || rank > 8 ||
        input.strides().size() != rank || out.strides().size() != rank ||
        input.shape() != out.shape() || input.dtype() != out.dtype() ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32) || !NonOverlappingOutput(out))
      return false;
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < rank; ++axis) {
      if (input.strides()[axis] < 0 || out.strides()[axis] < 0 ||
          (input.shape()[axis] && count >
              std::numeric_limits<std::size_t>::max() / input.shape()[axis]))
        return false;
      count *= input.shape()[axis];
    }
    std::size_t input_bytes = 0, out_bytes = 0;
    if (!SpanBytes(input, &input_bytes) || !SpanBytes(out, &out_bytes) ||
        input_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        out_bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) return false;
    const auto width = input.element_size();
    if ((input_bytes && reinterpret_cast<std::uintptr_t>(input.data()) % width) ||
        (out_bytes && reinterpret_cast<std::uintptr_t>(out.data()) % width))
      return false;
    const bool exact_inplace = input.data() == out.data() &&
        input.strides() == out.strides() && NonOverlappingOutput(input);
    if (Overlaps(input, input_bytes, out, out_bytes) && !exact_inplace)
      return false;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        current != input.device().index()) return false;
    if (input.dtype() != DataType::kFloat32) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, current) != cudaSuccess) return false;
      const int sm = 10 * prop.major + prop.minor;
      if (sm < (input.dtype() == DataType::kBFloat16 ? 80 : 53)) return false;
    }
    return true;
  }

  Operator(const Tensor input, Tensor out)
      : QuickGelu(checked(input, out), out) {}

  // QG1 has not yet passed standalone or installed-provider qualification.
  bool capture_safe() const override { return false; }

  void operator()(const Tensor input, Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        out.shape() != out_shape_ || out.strides() != out_strides_ ||
        input.dtype() != dtype_ || out.dtype() != dtype_ ||
        input.device() != device_ || out.device() != device_ ||
        !supports(input, out))
      throw std::invalid_argument("NVIDIA QuickGelu call metadata or alias changed after Make");
    std::size_t count = 1;
    nvidia_quick_gelu_detail::Layout layout{};
    layout.rank = static_cast<unsigned>(input.ndim());
    for (std::size_t axis = 0; axis < input.ndim(); ++axis) {
      layout.shape[axis] = input.shape()[axis];
      layout.input_strides[axis] = static_cast<std::size_t>(input.strides()[axis]);
      layout.out_strides[axis] = static_cast<std::size_t>(out.strides()[axis]);
      count *= input.shape()[axis];
    }
    if (!count) return;
    nvidia_quick_gelu_detail::Launch(input.data(), out.data(), dtype_, layout,
        count, static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  static Tensor checked(const Tensor input, Tensor out) {
    if (!supports(input, out))
      throw std::invalid_argument("unsupported NVIDIA QuickGelu metadata, device or alias");
    return input;
  }
};
}  // namespace infini::ops
#endif
