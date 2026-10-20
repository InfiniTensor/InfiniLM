#pragma once
#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "base/per_tensor_dequant_i8.h"
#include "base/per_tensor_quant_i8.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "native/cuda/nvidia/runtime_.h"
namespace infini::ops::nvidia_per_tensor_i8_detail {
struct Layout {
  std::int64_t ndim = 0, shape[4]{}, x_stride[4]{}, packed_stride[4]{};
};
void Quantize(const void*, std::int8_t*, float*, DataType, Layout, std::size_t,
              bool, cudaStream_t);
void Dequantize(void*, const std::int8_t*, const float*, DataType, Layout,
                std::size_t, cudaStream_t);
inline bool Supports(Tensor x, Tensor packed, Tensor scale,
                     const std::optional<Tensor> zero, bool quant,
                     bool is_static = true) {
  using namespace nvidia_binary_view;
  if (zero ||
      (x.dtype() != DataType::kFloat16 && x.dtype() != DataType::kFloat32 &&
       x.dtype() != DataType::kBFloat16) ||
      packed.dtype() != DataType::kInt8 ||
      scale.dtype() != DataType::kFloat32 || scale.numel() != 1 ||
      x.device().type() != Device::Type::kNvidia ||
      x.device() != packed.device() || x.device() != scale.device() ||
      x.shape() != packed.shape() || x.ndim() > 4 ||
      x.strides().size() != x.ndim() ||
      packed.strides().size() != packed.ndim() ||
      !(quant ? NonOverlappingOutput(packed) : NonOverlappingOutput(x)) ||
      (quant && !is_static && !NonOverlappingOutput(scale)))
    return false;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess || device != x.device().index())
    return false;
  if (x.dtype() == DataType::kBFloat16) {
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        major < 8)
      return false;
  }
  for (std::size_t d = 0; d < x.ndim(); ++d)
    if (x.strides()[d] < 0 || packed.strides()[d] < 0 ||
        x.shape()[d] > static_cast<std::size_t>(INT64_MAX))
      return false;
  std::size_t xb = 0, pb = 0, sb = 0;
  if (!SpanBytes(x, &xb) || !SpanBytes(packed, &pb) || !SpanBytes(scale, &sb) ||
      xb > INT64_MAX || pb > INT64_MAX || sb > INT64_MAX)
    return false;
  if ((xb && reinterpret_cast<std::uintptr_t>(x.data()) % x.element_size()) ||
      reinterpret_cast<std::uintptr_t>(scale.data()) % sizeof(float))
    return false;
  return !Overlaps(x, xb, packed, pb) &&
         !(quant && !is_static && Overlaps(x, xb, scale, sb)) &&
         !(quant ? Overlaps(packed, pb, scale, sb)
                 : Overlaps(x, xb, scale, sb));
}
inline Tensor Checked(Tensor x, Tensor packed, Tensor scale,
                      const std::optional<Tensor> zero, bool quant,
                      bool is_static = true) {
  if (!Supports(x, packed, scale, zero, quant, is_static))
    throw std::invalid_argument("unsupported per-tensor INT8 operands");
  return x;
}
struct Metadata {
  DataType dtype;
  Layout layout{};
  std::size_t count;
  Device device;
  Metadata(Tensor x, Tensor packed)
      : dtype(x.dtype()), count(x.numel()), device(x.device()) {
    layout.ndim = x.ndim();
    for (std::size_t d = 0; d < x.ndim(); ++d) {
      layout.shape[d] = x.shape()[d];
      layout.x_stride[d] = x.strides()[d];
      layout.packed_stride[d] = packed.strides()[d];
    }
  }
  bool Matches(Tensor x, Tensor packed) const {
    if (x.dtype() != dtype || x.device() != device ||
        x.ndim() != static_cast<std::size_t>(layout.ndim))
      return false;
    for (std::size_t d = 0; d < x.ndim(); ++d)
      if (x.shape()[d] != static_cast<std::size_t>(layout.shape[d]) ||
          x.strides()[d] != layout.x_stride[d] ||
          packed.strides()[d] != layout.packed_stride[d])
        return false;
    return true;
  }
};
}  // namespace infini::ops::nvidia_per_tensor_i8_detail
namespace infini::ops {
template <>
class Operator<PerTensorQuantI8, Device::Type::kNvidia, 0>
    : public PerTensorQuantI8 {
 public:
  static bool supports(const Tensor x, Tensor packed, Tensor scale,
                       const std::optional<Tensor> zero, const bool is_static) {
    return nvidia_per_tensor_i8_detail::Supports(x, packed, scale, zero, true,
                                                 is_static);
  }
  Operator(const Tensor x, Tensor packed, Tensor scale,
           const std::optional<Tensor> zero, const bool is_static)
      : PerTensorQuantI8(x, packed, scale, zero, is_static),
        meta_(nvidia_per_tensor_i8_detail::Checked(x, packed, scale, zero, true,
                                                   is_static),
              packed),
        is_static_(is_static) {}
  bool capture_safe() const override { return true; }
  void operator()(const Tensor x, Tensor packed, Tensor scale,
                  const std::optional<Tensor> zero,
                  const bool is_static) const override {
    if (!supports(x, packed, scale, zero, is_static) ||
        !meta_.Matches(x, packed) || is_static != is_static_)
      throw std::invalid_argument("unsupported per-tensor INT8 operands");
    nvidia_per_tensor_i8_detail::Quantize(
        x.data(), static_cast<std::int8_t*>(packed.data()),
        static_cast<float*>(scale.data()), meta_.dtype, meta_.layout,
        meta_.count, is_static,
        static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  nvidia_per_tensor_i8_detail::Metadata meta_;
  bool is_static_;
};
template <>
class Operator<PerTensorDequantI8, Device::Type::kNvidia, 0>
    : public PerTensorDequantI8 {
 public:
  static bool supports(Tensor x, const Tensor packed, const Tensor scale,
                       const std::optional<Tensor> zero) {
    return nvidia_per_tensor_i8_detail::Supports(x, packed, scale, zero, false);
  }
  Operator(Tensor x, const Tensor packed, const Tensor scale,
           const std::optional<Tensor> zero)
      : PerTensorDequantI8(x, packed, scale, zero),
        meta_(
            nvidia_per_tensor_i8_detail::Checked(x, packed, scale, zero, false),
            packed) {}
  bool capture_safe() const override { return true; }
  void operator()(Tensor x, const Tensor packed, const Tensor scale,
                  const std::optional<Tensor> zero) const override {
    if (!supports(x, packed, scale, zero) || !meta_.Matches(x, packed))
      throw std::invalid_argument("unsupported per-tensor INT8 operands");
    nvidia_per_tensor_i8_detail::Dequantize(
        x.data(), static_cast<const std::int8_t*>(packed.data()),
        static_cast<const float*>(scale.data()), meta_.dtype, meta_.layout,
        meta_.count, static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  nvidia_per_tensor_i8_detail::Metadata meta_;
};
}  // namespace infini::ops
