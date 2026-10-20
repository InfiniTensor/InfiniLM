#ifndef INFINI_OPS_CUDA_SEPARATE_TABLE_ROPE_LAYOUT_H_
#define INFINI_OPS_CUDA_SEPARATE_TABLE_ROPE_LAYOUT_H_
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace infini::ops::separate_rope_nvidia {
template <typename Tensor>
bool layout_supported(const Tensor& x, const Tensor& p, const Tensor& s,
                      const Tensor& c, bool neox, const Tensor& y) {
  if (x.numel() == 0) return true;  // operator returns without a launch.
  const auto rank = x.ndim();
  const auto batch = rank == 4 ? x.size(0) : 1;
  if (batch > 65535 || x.size(rank - 2) > 65535 ||
      x.size(rank - 3) > 2147483647u)
    return false;
  using D = std::decay_t<decltype(x.dtype())>;
  auto float_width = [](D dtype) -> std::size_t {
    return dtype == D::kFloat32 ? 4 : 2;  // base checks NVIDIA float domain.
  };
  auto aligned = [](const Tensor& t, std::size_t width) {
    // Pointer-free CanMake/Make metadata cannot establish address alignment;
    // actual nonnull addresses are checked again at prepared invocation.
    return !t.data() || reinterpret_cast<std::uintptr_t>(t.data()) % width == 0;
  };
  if (!aligned(x, float_width(x.dtype())) ||
      !aligned(y, float_width(y.dtype())) ||
      !aligned(s, float_width(s.dtype())) ||
      !aligned(c, float_width(c.dtype())) ||
      !aligned(p, p.dtype() == D::kInt64 ? 8 : 4))
    return false;
  // Existing interleaved kernel reinterprets half/bfloat16 pairs as half2 /
  // bfloat162. Every reachable row origin must be 4-byte aligned. Split-half
  // uses scalar loads/stores: 2-byte origins and odd outer strides are valid.
  if (!neox && x.dtype() != D::kFloat32) {
    if (!aligned(x, 4) || !aligned(y, 4)) return false;
    for (std::size_t axis = 0; axis + 1 < rank; ++axis) {
      if ((x.size(axis) > 1 && x.stride(axis) % 2 != 0) ||
          (y.size(axis) > 1 && y.stride(axis) % 2 != 0))
        return false;
    }
  }
  return true;
}
}  // namespace infini::ops::separate_rope_nvidia
#endif
