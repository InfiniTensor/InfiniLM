#ifndef INFINI_OPS_NVIDIA_MUL_SCALAR_H_
#define INFINI_OPS_NVIDIA_MUL_SCALAR_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "base/mul.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops::nvidia_mul_scalar_detail {
struct Layout {
  std::int64_t ndim = 0;
  std::int64_t shape[16]{};
  std::int64_t input_strides[16]{};
  std::int64_t out_strides[16]{};
  bool contiguous = false;
};

void Launch(const void* input, void* out, DataType dtype, double alpha,
            const Layout* layout, std::size_t count, cudaStream_t stream);
}  // namespace infini::ops::nvidia_mul_scalar_detail

namespace infini::ops {
class NvidiaScalarMul {
 public:
  static bool supports(const Tensor input, const double, Tensor out) {
    using namespace nvidia_binary_view;
    const auto ndim = input.shape().size();
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32 &&
         input.dtype() != DataType::kFloat64) ||
        ndim > 16 || out.shape() != input.shape() ||
        input.strides().size() != ndim || out.strides().size() != ndim ||
        !NonOverlappingOutput(out))
      return false;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != input.device().index())
      return false;
    if (input.dtype() == DataType::kBFloat16) {
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

    std::size_t count = 1;
    constexpr auto max_offset =
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    for (std::size_t d = 0; d < ndim; ++d) {
      const auto extent = input.shape()[d];
      if (extent > max_offset || input.strides()[d] < 0 ||
          out.strides()[d] < 0 ||
          static_cast<std::size_t>(input.strides()[d]) > max_offset ||
          static_cast<std::size_t>(out.strides()[d]) > max_offset ||
          (extent && count > max_offset / extent))
        return false;
      count *= extent;
    }
    std::size_t input_bytes = 0, out_bytes = 0;
    if (!SpanBytes(input, &input_bytes) || !SpanBytes(out, &out_bytes) ||
        input_bytes > max_offset || out_bytes > max_offset)
      return false;
    if (count &&
        (reinterpret_cast<std::uintptr_t>(input.data()) %
             input.element_size() ||
         reinterpret_cast<std::uintptr_t>(out.data()) % out.element_size()))
      return false;
    const bool exact_inplace = input.data() == out.data() &&
                               input.shape() == out.shape() &&
                               input.strides() == out.strides();
    return !Overlaps(input, input_bytes, out, out_bytes) || exact_inplace;
  }

  NvidiaScalarMul(const Tensor input, const double alpha, Tensor out)
      : input_meta_(Checked(input, alpha, out)),
        out_meta_(out),
        device_(input.device()),
        alpha_(alpha) {
    nvidia_binary_view::SpanBytes(input, &input_span_bytes_);
    nvidia_binary_view::SpanBytes(out, &out_span_bytes_);
    count_ = out.numel();
    layout_.ndim = static_cast<std::int64_t>(input.ndim());
    layout_.contiguous = input.IsContiguous() && out.IsContiguous();
    for (std::size_t d = 0; d < input.ndim(); ++d) {
      layout_.shape[d] = static_cast<std::int64_t>(input.shape()[d]);
      layout_.input_strides[d] = static_cast<std::int64_t>(input.strides()[d]);
      layout_.out_strides[d] = static_cast<std::int64_t>(out.strides()[d]);
    }
  }

  void Run(const Tensor input, const double alpha, Tensor out,
           cudaStream_t stream) const {
    if (!std::equal_to<Tensor>{}(input, input_meta_) ||
        !std::equal_to<Tensor>{}(out, out_meta_) ||
        std::memcmp(&alpha, &alpha_, sizeof(alpha)))
      throw std::invalid_argument(
          "NVIDIA scalar Mul metadata or alpha changed after Make");
    if (count_ == 0) return;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != device_.index() || !input.data() || !out.data() ||
        reinterpret_cast<std::uintptr_t>(input.data()) % input.element_size() ||
        reinterpret_cast<std::uintptr_t>(out.data()) % out.element_size() ||
        (nvidia_binary_view::Overlaps(input, input_span_bytes_, out,
                                      out_span_bytes_) &&
         !(input.data() == out.data() &&
           input_meta_.strides() == out_meta_.strides())))
      throw std::invalid_argument(
          "NVIDIA scalar Mul current device, pointer or alias changed after "
          "Make");
    nvidia_mul_scalar_detail::Launch(
        input.data(), out.data(), input_meta_.dtype(), alpha_,
        layout_.contiguous ? nullptr : &layout_, count_, stream);
  }

 private:
  static Tensor Checked(const Tensor input, const double alpha, Tensor out) {
    if (!supports(input, alpha, out))
      throw std::invalid_argument(
          "unsupported NVIDIA scalar Mul metadata, device or alias");
    return input;
  }
  nvidia_mul_scalar_detail::Layout layout_{};
  const Tensor input_meta_, out_meta_;
  Device device_;
  double alpha_;
  std::size_t input_span_bytes_ = 0, out_span_bytes_ = 0, count_ = 0;
};
}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_MUL_SCALAR_KERNEL_H_
