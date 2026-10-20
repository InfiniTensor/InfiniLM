#ifndef INFINI_OPS_HYGON_BINARY_VIEW_CAPTURE_H_
#define INFINI_OPS_HYGON_BINARY_VIEW_CAPTURE_H_

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "tensor.h"

namespace infini::ops::hygon_binary_view {

inline bool SpanBytes(const Tensor tensor, std::size_t* bytes) {
  if (tensor.shape().size() != tensor.strides().size() || !tensor.element_size())
    return false;
  for (auto dim : tensor.shape()) if (!dim) { *bytes = 0; return true; }
  std::size_t last = 0;
  for (std::size_t i = 0; i < tensor.ndim(); ++i) {
    if (tensor.strides()[i] < 0) return false;
    const auto stride = static_cast<std::size_t>(tensor.strides()[i]);
    const auto extent = tensor.shape()[i] - 1;
    if (extent && stride > (std::numeric_limits<std::size_t>::max() - last) / extent)
      return false;
    last += extent * stride;
  }
  if (last == std::numeric_limits<std::size_t>::max() ||
      last + 1 > std::numeric_limits<std::size_t>::max() / tensor.element_size())
    return false;
  *bytes = (last + 1) * tensor.element_size();
  return !*bytes || tensor.data();
}

inline bool NonOverlappingOutput(const Tensor out) {
  if (out.shape().size() != out.strides().size()) return false;
  if (out.HasBroadcastDim()) return false;
  for (auto dim : out.shape()) if (!dim) return true;
  std::vector<std::pair<std::size_t, std::size_t>> dimensions;
  for (std::size_t i = 0; i < out.ndim(); ++i) {
    if (out.strides()[i] < 0) return false;
    if (out.shape()[i] > 1)
      dimensions.emplace_back(static_cast<std::size_t>(out.strides()[i]),
                              out.shape()[i]);
  }
  std::sort(dimensions.begin(), dimensions.end());
  std::size_t span = 1;
  for (const auto& [stride, extent] : dimensions) {
    if (stride < span ||
        extent - 1 > (std::numeric_limits<std::size_t>::max() - span) / stride)
      return false;
    span += (extent - 1) * stride;
  }
  return true;
}

inline bool Overlaps(const Tensor a, std::size_t a_bytes,
                     const Tensor b, std::size_t b_bytes) {
  if (!a_bytes || !b_bytes) return false;
  const auto lhs = reinterpret_cast<std::uintptr_t>(a.data());
  const auto rhs = reinterpret_cast<std::uintptr_t>(b.data());
  return lhs <= rhs ? rhs - lhs < a_bytes : lhs - rhs < b_bytes;
}

inline bool PointersSupported(const Tensor input, const Tensor other, Tensor out) {
  if (input.device() != other.device() || input.device() != out.device())
    return false;
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess ||
      current_device != input.device().index()) return false;
  std::size_t input_bytes, other_bytes, out_bytes;
  if (!SpanBytes(input, &input_bytes) || !SpanBytes(other, &other_bytes) ||
      !SpanBytes(out, &out_bytes)) return false;
  const auto safe_alias = [&](const Tensor source, std::size_t bytes) {
    return !Overlaps(source, bytes, out, out_bytes) ||
        (source.data() == out.data() && source.shape() == out.shape() &&
         source.strides() == out.strides());
  };
  return safe_alias(input, input_bytes) && safe_alias(other, other_bytes);
}

inline bool Supported(const Tensor input, const Tensor other, Tensor out) {
  if (input.shape().size() != input.strides().size() ||
      other.shape().size() != other.strides().size() ||
      out.shape().size() != out.strides().size() ||
      input.device().type() != Device::Type::kHygon ||
      input.dtype() != other.dtype() || input.dtype() != out.dtype() ||
      input.ndim() > out.ndim() || other.ndim() > out.ndim() ||
      !NonOverlappingOutput(out)) return false;
  for (std::size_t i = 0; i < out.ndim(); ++i) {
    const auto a = i < out.ndim() - input.ndim() ? 1 :
        input.size(i + input.ndim() - out.ndim());
    const auto b = i < out.ndim() - other.ndim() ? 1 :
        other.size(i + other.ndim() - out.ndim());
    // Current Add/Mul base constructors do not implement the valid 0/1
    // empty broadcast. Keep this provider fail-closed until that is fixed.
    if ((a == 0 && b == 1) || (a == 1 && b == 0) ||
        (a != b && a != 1 && b != 1) || out.size(i) != std::max(a, b))
      return false;
  }
  return PointersSupported(input, other, out);
}

struct Metadata {
  Tensor::Shape input_shape, other_shape, out_shape;
  Tensor::Strides input_strides, other_strides, out_strides;
  DataType dtype;
  Device device;

  Metadata(const Tensor input, const Tensor other, const Tensor out)
      : input_shape(input.shape()), other_shape(other.shape()),
        out_shape(out.shape()), input_strides(input.strides()),
        other_strides(other.strides()), out_strides(out.strides()),
        dtype(out.dtype()), device(out.device()) {}

  bool Matches(const Tensor input, const Tensor other, Tensor out) const {
    return input.shape() == input_shape && other.shape() == other_shape &&
        out.shape() == out_shape && input.strides() == input_strides &&
        other.strides() == other_strides && out.strides() == out_strides &&
        input.dtype() == dtype && other.dtype() == dtype &&
        out.dtype() == dtype && out.device() == device &&
        PointersSupported(input, other, out);
  }
};

}  // namespace infini::ops::hygon_binary_view

#endif
