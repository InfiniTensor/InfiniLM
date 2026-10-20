#ifndef INFINI_OPS_CPU_DENSE_VIEW_H_
#define INFINI_OPS_CPU_DENSE_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensor.h"

namespace infini::ops::cpu_dense_view {

// Storage checks only; callers retain their dtype, shape and device contracts.
inline bool ContiguousBytes(const Tensor t, std::size_t* bytes) {
  if (t.shape().size() != t.strides().size()) return false;
  const auto limit =
      static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
      t.element_size();
  bool empty = false;
  for (std::size_t axis = 0; axis < t.ndim(); ++axis) {
    if (t.stride(axis) < 0 || t.size(axis) > limit) return false;
    empty = empty || t.size(axis) == 0;
  }
  if (empty) {
    *bytes = 0;
    return true;
  }
  std::size_t count = 1;
  for (std::size_t axis = t.ndim(); axis > 0; --axis) {
    const auto size = t.size(axis - 1);
    if ((size > 1 &&
         t.stride(axis - 1) != static_cast<std::ptrdiff_t>(count)) ||
        size > limit / count)
      return false;
    count *= size;
  }
  *bytes = count * t.element_size();
  return true;
}

inline bool ValidAddress(const Tensor t, std::size_t bytes) {
  if (bytes == 0) return true;
  const auto start = reinterpret_cast<std::uintptr_t>(t.data());
  return start && start % t.element_size() == 0 &&
         start <= std::numeric_limits<std::uintptr_t>::max() - bytes;
}

inline bool DisjointOrSameContiguous(const Tensor input, Tensor out) {
  std::size_t input_bytes = 0, out_bytes = 0;
  if (!ContiguousBytes(input, &input_bytes) ||
      !ContiguousBytes(out, &out_bytes) || !ValidAddress(input, input_bytes) ||
      !ValidAddress(out, out_bytes))
    return false;
  const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
  const auto op = reinterpret_cast<std::uintptr_t>(out.data());
  const bool overlap =
      input_bytes && out_bytes && ip < op + out_bytes && op < ip + input_bytes;
  return !overlap || (ip == op && input.strides() == out.strides());
}

}  // namespace infini::ops::cpu_dense_view

#endif  // INFINI_OPS_CPU_DENSE_VIEW_H_
