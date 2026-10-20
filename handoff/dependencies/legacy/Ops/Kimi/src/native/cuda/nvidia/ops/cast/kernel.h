#ifndef INFINI_OPS_NVIDIA_CAST_KERNEL_H_
#define INFINI_OPS_NVIDIA_CAST_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "base/cast.h"
#include "native/cuda/nvidia/caster.cuh"

namespace infini::ops {
namespace nvidia_cast_detail {
template <typename InT, typename OutT>
__global__ void Convert(const InT* input, OutT* out, std::size_t first,
                       std::size_t count) {
  const auto local = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (local < count)
    out[first + local] = Caster<Device::Type::kNvidia>::template Cast<OutT>(input[first + local]);
}
}  // namespace nvidia_cast_detail

template <>
class Operator<Cast, Device::Type::kNvidia> : public Cast {
 public:
  static bool supports(const Tensor input, Tensor out) {
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != out.device() || input.shape() != out.shape() ||
        !SupportedDtype(input.dtype()) || !SupportedDtype(out.dtype())) return false;
    std::size_t input_bytes = 0, out_bytes = 0;
    if (!ContiguousBytes(input, &input_bytes) ||
        !ContiguousBytes(out, &out_bytes) ||
        !ValidAddress(input, input_bytes) || !ValidAddress(out, out_bytes)) return false;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != input.device().index())
      return false;
    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    const bool overlap = input_bytes && out_bytes &&
        ip < op + out_bytes && op < ip + input_bytes;
    return !overlap || SameView(input, out);
  }

  Operator(const Tensor input, Tensor out)
      : Cast{Checked(input, out), out}, device_{input.device()} {}

  bool capture_safe() const override { return false; }

  void operator()(const Tensor input, Tensor out) const override {
    if (input.dtype() != input_dtype_ || out.dtype() != out_dtype_ ||
        input.shape() != input_shape_ || out.shape() != out_shape_ ||
        input.strides() != input_strides_ || out.strides() != out_strides_ ||
        input.device() != device_ || !supports(input, out))
      throw std::invalid_argument("NVIDIA Cast: call metadata or alias changed after Make");
    if (output_size_ == 0 || SameView(input, out)) return;
    const auto stream = static_cast<cudaStream_t>(stream_);
    DispatchFunc<List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>,
                 List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        {static_cast<int64_t>(input_dtype_), static_cast<int64_t>(out_dtype_)},
        [&](auto list_tag) {
          using InT = TypeMapType<Device::Type::kNvidia, ListGet<0>(list_tag)>;
          using OutT = TypeMapType<Device::Type::kNvidia, ListGet<1>(list_tag)>;
          constexpr std::size_t cap = 65535u * 256u;
          for (std::size_t first = 0; first < output_size_;) {
            const auto count = std::min(cap, output_size_ - first);
            const auto blocks = static_cast<unsigned>((count + 255) / 256);
            nvidia_cast_detail::Convert<InT, OutT><<<blocks, 256, 0, stream>>>(
                static_cast<const InT*>(input.data()), static_cast<OutT*>(out.data()),
                first, count);
            const auto status = cudaGetLastError();
            if (status != cudaSuccess)
              throw std::runtime_error(std::string("NVIDIA Cast: launch failed: ") +
                                       cudaGetErrorString(status));
            first += count;
          }
        }, "NVIDIA Cast");
  }

 private:
  static bool SupportedDtype(DataType type) {
    return type == DataType::kFloat32 || type == DataType::kFloat16 ||
           type == DataType::kBFloat16;
  }

  static std::size_t Width(DataType type) {
    return type == DataType::kFloat32 ? 4u : 2u;
  }

  static bool ContiguousBytes(const Tensor t, std::size_t* bytes) {
    if (t.shape().size() != t.strides().size()) return false;
    bool empty = false;
    for (std::size_t axis = 0; axis < t.ndim(); ++axis) {
      if (t.stride(axis) < 0) return false;
      empty = empty || t.size(axis) == 0;
    }
    if (empty) { *bytes = 0; return true; }
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
                       Width(t.dtype());
    std::size_t count = 1;
    for (std::size_t axis = t.ndim(); axis > 0; --axis) {
      const auto size = t.size(axis - 1);
      if (size > 1 && t.stride(axis - 1) != static_cast<std::ptrdiff_t>(count))
        return false;
      if (size > limit / count) return false;
      count *= size;
    }
    *bytes = count * Width(t.dtype());
    return true;
  }

  static bool ValidAddress(const Tensor t, std::size_t bytes) {
    if (bytes == 0) return true;
    const auto start = reinterpret_cast<std::uintptr_t>(t.data());
    return start && start % Width(t.dtype()) == 0 &&
        start <= std::numeric_limits<std::uintptr_t>::max() - bytes;
  }

  static bool SameView(const Tensor input, const Tensor out) {
    return input.dtype() == out.dtype() && input.data() == out.data() &&
        input.shape() == out.shape() && input.strides() == out.strides();
  }

  static Tensor Checked(const Tensor input, const Tensor out) {
    if (!supports(input, out))
      throw std::invalid_argument("NVIDIA Cast: unsupported dtype/device/layout or alias");
    return input;
  }

  Device device_;
};
}  // namespace infini::ops
#endif
