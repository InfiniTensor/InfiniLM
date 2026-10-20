#ifndef INFINI_OPS_NVIDIA_PAD_CONSTANT_KERNEL_H_
#define INFINI_OPS_NVIDIA_PAD_CONSTANT_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "base/pad_constant.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops::nvidia_pad_constant_detail {
struct Layout {
  std::int64_t ndim = 0;
  std::int64_t input_shape[16]{};
  std::int64_t output_shape[16]{};
  std::int64_t input_strides[16]{};
  std::int64_t output_strides[16]{};
  std::int64_t left_padding[16]{};
};

void Launch(const void* input, void* out, DataType dtype, double value,
            Layout layout, std::size_t output_count, cudaStream_t stream);
}  // namespace infini::ops::nvidia_pad_constant_detail

namespace infini::ops {
template <>
class Operator<PadConstant, Device::Type::kNvidia, 0>
    : public PadConstant {
 public:
  static bool supports(const Tensor input,
                       const std::vector<std::int64_t> padding,
                       const double, Tensor out) {
    using namespace nvidia_binary_view;
    const auto ndim = input.shape().size();
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32 &&
         input.dtype() != DataType::kFloat64) ||
        ndim == 0 || ndim > 16 || out.shape().size() != ndim ||
        input.strides().size() != ndim || out.strides().size() != ndim ||
        padding.empty() || padding.size() % 2 || padding.size() > 2 * ndim ||
        !NonOverlappingOutput(out))
      return false;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != input.device().index()) return false;
    if (input.dtype() == DataType::kBFloat16) {
      // Device architecture is immutable, while supports() is on the Call hot
      // path. Keep the metadata-only qualification local to each host thread.
      static thread_local int cached_device = -1;
      static thread_local bool cached_bf16 = false;
      if (cached_device != current_device) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   current_device) != cudaSuccess)
          return false;
        cached_bf16 = major >= 8;
        cached_device = current_device;
      }
      if (!cached_bf16) return false;
    }

    const auto max_i64 = static_cast<std::uint64_t>(
        std::numeric_limits<std::int64_t>::max());
    std::size_t input_count = 1, output_count = 1;
    bool any_padding = false;
    for (std::size_t d = 0; d < ndim; ++d) {
      const auto in_size = input.shape()[d];
      const auto out_size = out.shape()[d];
      const auto in_stride = input.strides()[d];
      const auto out_stride = out.strides()[d];
      const auto pair = ndim - 1 - d;
      const auto left = pair < padding.size() / 2 ? padding[2 * pair] : 0;
      const auto right = pair < padding.size() / 2 ? padding[2 * pair + 1] : 0;
      if (left < 0 || right < 0 || in_size > max_i64 || out_size > max_i64 ||
          in_stride < 0 || out_stride < 0 ||
          static_cast<std::uint64_t>(in_stride) > max_i64 ||
          static_cast<std::uint64_t>(out_stride) > max_i64 ||
          static_cast<std::uint64_t>(left) > max_i64 - in_size ||
          static_cast<std::uint64_t>(right) >
              max_i64 - in_size - static_cast<std::uint64_t>(left) ||
          out_size != in_size + static_cast<std::size_t>(left) +
                          static_cast<std::size_t>(right) ||
          (in_size && input_count > std::numeric_limits<std::size_t>::max() / in_size) ||
          (out_size && output_count > std::numeric_limits<std::size_t>::max() / out_size))
        return false;
      input_count *= in_size;
      output_count *= out_size;
      any_padding |= left != 0 || right != 0;
    }
    std::size_t input_bytes = 0, output_bytes = 0;
    if (!SpanBytes(input, &input_bytes) || !SpanBytes(out, &output_bytes))
      return false;
    const bool exact_inplace = !any_padding && input.data() == out.data() &&
        input.shape() == out.shape() && input.strides() == out.strides();
    return !Overlaps(input, input_bytes, out, output_bytes) || exact_inplace;
  }

  Operator(const Tensor input, const std::vector<std::int64_t> padding,
           const double value, Tensor out)
      : PadConstant(Checked(input, padding, value, out), padding, value, out) {
    BuildLayout(input, padding, out);
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input,
                  const std::vector<std::int64_t> padding,
                  const double value, Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        out.shape() != out_shape_ || out.strides() != out_strides_ ||
        input.dtype() != dtype_ || out.dtype() != dtype_ ||
        input.device() != device_ || out.device() != device_ ||
        padding != padding_ || std::memcmp(&value, &value_, sizeof(value)) ||
        !supports(input, padding, value, out))
      throw std::invalid_argument("NVIDIA PadConstant metadata or alias changed after Make");
    if (out.numel() == 0 || (input.data() == out.data() &&
        input.shape() == out.shape() && input.strides() == out.strides()))
      return;
    nvidia_pad_constant_detail::Launch(
        input.data(), out.data(), dtype_, value_, layout_, out.numel(),
        static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  static Tensor Checked(const Tensor input,
                        const std::vector<std::int64_t>& padding,
                        double value, Tensor out) {
    if (!supports(input, padding, value, out))
      throw std::invalid_argument("unsupported NVIDIA PadConstant metadata, device or alias");
    return input;
  }
  void BuildLayout(const Tensor input,
                   const std::vector<std::int64_t>& padding, Tensor out) {
    layout_.ndim = static_cast<std::int64_t>(input.ndim());
    for (std::size_t d = 0; d < input.ndim(); ++d) {
      const auto pair = input.ndim() - 1 - d;
      layout_.input_shape[d] = static_cast<std::int64_t>(input.shape()[d]);
      layout_.output_shape[d] = static_cast<std::int64_t>(out.shape()[d]);
      layout_.input_strides[d] = static_cast<std::int64_t>(input.strides()[d]);
      layout_.output_strides[d] = static_cast<std::int64_t>(out.strides()[d]);
      layout_.left_padding[d] = pair < padding.size() / 2 ? padding[2 * pair] : 0;
    }
  }
  nvidia_pad_constant_detail::Layout layout_{};
};
}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_PAD_CONSTANT_KERNEL_H_
