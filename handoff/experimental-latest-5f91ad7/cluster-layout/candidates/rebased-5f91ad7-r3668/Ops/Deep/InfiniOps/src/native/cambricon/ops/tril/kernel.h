#ifndef INFINI_OPS_CAMBRICON_TRIL_KERNEL_H_
#define INFINI_OPS_CAMBRICON_TRIL_KERNEL_H_
#include <climits>
#include <cstdint>
#include <stdexcept>

#include "base/tril.h"
#include "native/cambricon/common.h"
#include "native/cambricon/dense_triangle.h"
namespace infini::ops {
template <>
class Operator<Tril, Device::Type::kCambricon> : public Tril {
 public:
  static bool supports(const Tensor x, int64_t, const Tensor y) {
    if (x.device().type() != Device::Type::kCambricon ||
        x.device().index() < 0 || x.dtype() != DataType::kFloat32 ||
        y.dtype() != x.dtype() || x.shape() != y.shape() ||
        x.device() != y.device() || x.ndim() < 2 || !x.IsContiguous() ||
        !y.IsContiguous() || !x.data() || !y.data())
      return false;
    size_t count = 1;
    for (auto n : x.shape()) {
      if (!n || n > INT_MAX || count > size_t(INT_MAX) / n) return false;
      count *= n;
    }
    const auto a = reinterpret_cast<uintptr_t>(x.data()),
               b = reinterpret_cast<uintptr_t>(y.data());
    if (a % sizeof(float) || b % sizeof(float)) return false;
    return a == b || (a > b ? a - b >= count * sizeof(float)
                            : b - a >= count * sizeof(float));
  }
  Operator(const Tensor x, int64_t diagonal, Tensor y) : Tril(x, diagonal, y) {}
  void operator()(const Tensor x, int64_t diagonal, Tensor y) const override {
    if (!supports(x, diagonal, y) || x.shape() != input_shape_ ||
        x.strides() != input_strides_ || x.dtype() != input_type_ ||
        y.shape() != out_shape_ || y.strides() != out_strides_ ||
        y.dtype() != out_type_ || x.device().index() != device_index_)
      throw std::invalid_argument(
          "Cambricon Tril prepared FP32 metadata changed");
    DenseTriangleLaunch(static_cast<cnrtQueue_t>(stream_), x.data(), y.data(),
                        y.numel(), y.size(-2), y.size(-1), diagonal, true,
                        sizeof(float));
    CNRT_CHECK(cnrtGetLastError());
  }
};
}  // namespace infini::ops
#endif
