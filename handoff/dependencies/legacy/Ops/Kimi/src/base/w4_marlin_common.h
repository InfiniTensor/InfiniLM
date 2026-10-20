#pragma once
#include "operator.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infini::ops::w4 {
inline void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(message);
}
inline std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("W4 size overflow");
  return a * b;
}
inline bool fp16(const Tensor& t) {
  return t.dtype() == DataType::kFloat16 || t.dtype() == DataType::kBFloat16;
}
inline void shape(const Tensor& t, const Tensor::Shape& s, DataType type) {
  require(t.shape() == s && t.dtype() == type && t.IsContiguous(),
          "W4 requires the exact contiguous shape and dtype");
}
inline void device(const Tensor& ref, const Tensor& t) {
  require(ref.device() == t.device(), "W4 requires one device");
}
inline void dims(int64_t k, int64_t n) {
  require(k > 0 && n > 0 && k % 128 == 0 && n % 64 == 0 &&
          k <= std::numeric_limits<int>::max() && n <= std::numeric_limits<int>::max() / 2,
          "W4 Marlin requires K%128=0, N%64=0 and bounded dimensions");
  require(mul(static_cast<std::size_t>(k), static_cast<std::size_t>(n))/8 <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()),
          "W4 packed indexing exceeds original kernel int32 limits");
}
inline int64_t groups(int64_t k, int64_t group) {
  require(group == -1 || group == 32 || group == 64 || group == 128,
          "W4 supports groups -1,32,64,128");
  require(group == -1 || k % group == 0, "W4 group must divide K");
  return group == -1 ? 1 : k / group;
}
inline std::size_t span(const Tensor& t) {
  if (!t.numel()) return 0;
  std::size_t extent = 1;
  for (int64_t i = 0; i < static_cast<int64_t>(t.ndim()); ++i) {
    require(t.stride(i) >= 0, "W4 negative stride is unsupported");
    const auto part = mul(t.size(i) - 1, static_cast<std::size_t>(t.stride(i)));
    if (part > std::numeric_limits<std::size_t>::max() - extent)
      throw std::overflow_error("W4 tensor span overflow");
    extent += part;
  }
  return mul(extent, t.element_size());
}
inline void disjoint(const void* a, std::size_t ab, const void* b, std::size_t bb) {
  if (!ab || !bb || !a || !b) return; // Make may use address-free metadata.
  const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
  require(ab <= std::numeric_limits<std::uintptr_t>::max() - x &&
          bb <= std::numeric_limits<std::uintptr_t>::max() - y,
          "W4 address range overflow");
  require(x + ab <= y || y + bb <= x, "W4 writable storage overlaps an input/output");
}
inline void disjoint(const Tensor& a, const Tensor& b) {
  disjoint(a.data(), span(a), b.data(), span(b));
}
inline void address(const Tensor& t, std::size_t alignment) {
  if (!t.numel()) return;
  require(t.data() && reinterpret_cast<std::uintptr_t>(t.data()) % alignment == 0,
          "W4 device pointer is null or misaligned");
}
} // namespace infini::ops::w4
