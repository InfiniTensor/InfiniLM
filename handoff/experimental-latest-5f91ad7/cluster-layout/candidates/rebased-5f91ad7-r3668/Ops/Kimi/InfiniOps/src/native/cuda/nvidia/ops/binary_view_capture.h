#ifndef INFINI_OPS_NVIDIA_BINARY_VIEW_CAPTURE_H_
#define INFINI_OPS_NVIDIA_BINARY_VIEW_CAPTURE_H_

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "tensor.h"

namespace infini::ops::nvidia_binary_view {

inline bool SpanBytes(const Tensor tensor, std::size_t* bytes) {
  if (tensor.shape().size() != tensor.strides().size() ||
      !tensor.element_size())
    return false;
  for (auto dim : tensor.shape())
    if (!dim) {
      *bytes = 0;
      return true;
    }
  std::size_t last = 0;
  for (std::size_t i = 0; i < tensor.ndim(); ++i) {
    if (tensor.strides()[i] < 0) return false;
    const auto stride = static_cast<std::size_t>(tensor.strides()[i]);
    const auto extent = tensor.shape()[i] - 1;
    if (extent &&
        stride > (std::numeric_limits<std::size_t>::max() - last) / extent)
      return false;
    last += extent * stride;
  }
  if (last == std::numeric_limits<std::size_t>::max() ||
      last + 1 >
          std::numeric_limits<std::size_t>::max() / tensor.element_size())
    return false;
  *bytes = (last + 1) * tensor.element_size();
  return !*bytes || tensor.data();
}

inline bool NonOverlappingOutput(const Tensor out) {
  if (out.shape().size() != out.strides().size()) return false;
  if (out.HasBroadcastDim()) return false;
  for (auto dim : out.shape())
    if (!dim) return true;
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

inline bool Overlaps(const Tensor a, std::size_t a_bytes, const Tensor b,
                     std::size_t b_bytes) {
  if (!a_bytes || !b_bytes) return false;
  const auto lhs = reinterpret_cast<std::uintptr_t>(a.data());
  const auto rhs = reinterpret_cast<std::uintptr_t>(b.data());
  return lhs <= rhs ? rhs - lhs < a_bytes : lhs - rhs < b_bytes;
}

}  // namespace infini::ops::nvidia_binary_view

#endif
