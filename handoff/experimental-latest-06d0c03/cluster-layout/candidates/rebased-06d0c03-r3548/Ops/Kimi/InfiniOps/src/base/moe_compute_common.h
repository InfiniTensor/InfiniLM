#pragma once
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include "operator.h"
namespace infini::ops::moe_detail {
inline void Require(bool ok) {
  if (!ok)
    throw std::invalid_argument(
        "MoE metadata, parameter, pointer, or alias contract violated");
}
inline bool Reduced(DataType d) {
  return d == DataType::kFloat16 || d == DataType::kBFloat16;
}
inline size_t Product(size_t a, size_t b) {
  Require(!b || a <= std::numeric_limits<size_t>::max() / b);
  return a * b;
}
inline void Bound(size_t n) {
  Require(n <= size_t(std::numeric_limits<int32_t>::max()));
}
inline bool Overlap(const Tensor a, const Tensor b) {
  const size_t an = Product(a.numel(), a.element_size()),
               bn = Product(b.numel(), b.element_size());
  if (!an || !bn) return false;
  const uintptr_t ap = reinterpret_cast<uintptr_t>(a.data()),
                  bp = reinterpret_cast<uintptr_t>(b.data());
  return ap <= bp ? bp - ap < an : ap - bp < bn;
}
inline void Views(const std::vector<Tensor>& v, size_t output_count,
                  bool output_hidden_alias = false) {
  Require(!v.empty());
  for (size_t i = 0; i < v.size(); ++i) {
    Require(v[i].device() == v[0].device() && v[i].IsContiguous());
    Product(v[i].numel(), v[i].element_size());
    Require(!v[i].numel() || v[i].data());
    Require(!v[i].numel() ||
            reinterpret_cast<uintptr_t>(v[i].data()) % v[i].element_size() ==
                0);
    for (size_t j = 0; j < i && j < output_count; ++j) {
      if (!(output_hidden_alias && j == 0 && i == 1))
        Require(!Overlap(v[i], v[j]));
    }
  }
}
inline void Same(const std::vector<Tensor>& a, const std::vector<Tensor>& b) {
  Require(a.size() == b.size());
  const std::equal_to<Tensor> eq;
  for (size_t i = 0; i < a.size(); ++i) Require(eq(a[i], b[i]));
}
inline void Workspace(void* p, size_t available, size_t required,
                      const std::vector<Tensor>& v) {
  Require(p && available >= required &&
          reinterpret_cast<uintptr_t>(p) % 256 == 0);
  const auto ap = reinterpret_cast<uintptr_t>(p);
  for (const auto& t : v) {
    const auto bp = reinterpret_cast<uintptr_t>(t.data());
    const auto bytes = Product(t.numel(), t.element_size());
    if (bytes) Require(ap <= bp ? bp - ap >= required : ap - bp >= bytes);
  }
}
}  // namespace infini::ops::moe_detail
