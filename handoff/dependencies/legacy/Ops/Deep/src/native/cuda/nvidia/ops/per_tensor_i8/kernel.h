#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "base/per_tensor_quant_i8.h"
#include "base/per_tensor_dequant_i8.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "native/cuda/nvidia/runtime_.h"
namespace infini::ops::nvidia_per_tensor_i8_detail {
struct Layout {
  std::int64_t ndim = 0, shape[4]{}, x_stride[4]{}, packed_stride[4]{};
};
void Quantize(const void*, std::int8_t*, float*, DataType, Layout,
              std::size_t, bool, cudaStream_t);
void Dequantize(void*, const std::int8_t*, const float*, DataType, Layout,
                std::size_t, cudaStream_t);
inline bool Supports(Tensor x, Tensor packed, Tensor scale,
                     const std::optional<Tensor> zero) {
  using namespace nvidia_binary_view;
  if (zero || (x.dtype() != DataType::kFloat16 && x.dtype() != DataType::kFloat32 &&
               x.dtype() != DataType::kBFloat16) || packed.dtype() != DataType::kInt8 ||
      scale.dtype() != DataType::kFloat32 || scale.numel() != 1 ||
      x.device().type() != Device::Type::kNvidia || x.device() != packed.device() ||
      x.device() != scale.device() || x.shape() != packed.shape() ||
      x.ndim() > 4 || !NonOverlappingOutput(x) ||
      !NonOverlappingOutput(packed)) return false;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess || device != x.device().index())
    return false;
  if (x.dtype() == DataType::kBFloat16) {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)
        != cudaSuccess || major < 8) return false;
  }
  for (std::size_t d = 0; d < x.ndim(); ++d)
    if (x.strides()[d] < 0 || packed.strides()[d] < 0 ||
        x.shape()[d] > static_cast<std::size_t>(INT64_MAX)) return false;
  std::size_t xb = 0, pb = 0, sb = 0;
  if (!SpanBytes(x, &xb) || !SpanBytes(packed, &pb) || !SpanBytes(scale, &sb) ||
      xb > INT64_MAX || pb > INT64_MAX || sb > INT64_MAX) return false;
  if ((xb && reinterpret_cast<std::uintptr_t>(x.data()) % x.element_size()) ||
      reinterpret_cast<std::uintptr_t>(scale.data()) % sizeof(float)) return false;
  return !Overlaps(x, xb, packed, pb) && !Overlaps(x, xb, scale, sb) &&
         !Overlaps(packed, pb, scale, sb);
}
struct Metadata {
  DataType dtype;
  Layout layout{};
  std::size_t count;
  Metadata(Tensor x, Tensor packed)
      : dtype(x.dtype()), count(x.numel()) {
    layout.ndim = x.ndim();
    for (std::size_t d = 0; d < x.ndim(); ++d) {
      layout.shape[d] = x.shape()[d]; layout.x_stride[d] = x.strides()[d];
      layout.packed_stride[d] = packed.strides()[d];
    }
  }
};
}
namespace infini::ops {
template <> class Operator<PerTensorQuantI8, Device::Type::kNvidia, 0>
    : public PerTensorQuantI8 {
 public:
  static bool supports(const Tensor x, Tensor packed, Tensor scale,
                       const std::optional<Tensor> zero, const bool) {
    return nvidia_per_tensor_i8_detail::Supports(x, packed, scale, zero);
  }
  Operator(const Tensor x, Tensor packed, Tensor scale,
           const std::optional<Tensor> zero, const bool is_static)
      : PerTensorQuantI8(x, packed, scale, zero, is_static),
        meta_(x, packed) {}
  bool capture_safe() const override { return true; }
  void operator()(const Tensor x, Tensor packed, Tensor scale,
                  const std::optional<Tensor> zero, const bool is_static) const override {
    if (!supports(x, packed, scale, zero, is_static))
      throw std::invalid_argument("unsupported per-tensor INT8 operands");
    nvidia_per_tensor_i8_detail::Quantize(x.data(),
        static_cast<std::int8_t*>(packed.data()), static_cast<float*>(scale.data()),
        meta_.dtype, meta_.layout, meta_.count, is_static,
        static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }
 private:
  nvidia_per_tensor_i8_detail::Metadata meta_;
};
template <> class Operator<PerTensorDequantI8, Device::Type::kNvidia, 0>
    : public PerTensorDequantI8 {
 public:
  static bool supports(Tensor x, const Tensor packed, const Tensor scale,
                       const std::optional<Tensor> zero) {
    return nvidia_per_tensor_i8_detail::Supports(x, packed, scale, zero);
  }
  Operator(Tensor x, const Tensor packed, const Tensor scale,
           const std::optional<Tensor> zero)
      : PerTensorDequantI8(x, packed, scale, zero), meta_(x, packed) {}
  bool capture_safe() const override { return true; }
  void operator()(Tensor x, const Tensor packed, const Tensor scale,
                  const std::optional<Tensor> zero) const override {
    if (!supports(x, packed, scale, zero))
      throw std::invalid_argument("unsupported per-tensor INT8 operands");
    nvidia_per_tensor_i8_detail::Dequantize(x.data(),
        static_cast<const std::int8_t*>(packed.data()),
        static_cast<const float*>(scale.data()), meta_.dtype, meta_.layout,
        meta_.count, static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }
 private:
  nvidia_per_tensor_i8_detail::Metadata meta_;
};
}
