#ifndef INFINI_OPS_CPU_MUL_SCALAR_MUL_SCALAR_H_
#define INFINI_OPS_CPU_MUL_SCALAR_MUL_SCALAR_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "base/mul_scalar.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<MulScalar, Device::Type::kCpu, 0> : public MulScalar,
                                                 Caster<Device::Type::kCpu> {
 public:
  static bool supports(const Tensor input, const double, Tensor out) {
    if (input.device().type() != Device::Type::kCpu ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        input.shape() != out.shape() || !SupportedDtype(input.dtype()))
      return false;
    std::size_t input_bytes = 0, out_bytes = 0;
    if (!ContiguousBytes(input, &input_bytes) ||
        !ContiguousBytes(out, &out_bytes) ||
        !ValidAddress(input, input_bytes) || !ValidAddress(out, out_bytes))
      return false;
    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    const bool overlap = input_bytes && out_bytes &&
        ip < op + out_bytes && op < ip + input_bytes;
    return !overlap || (ip == op && input.strides() == out.strides());
  }

  Operator(const Tensor input, const double alpha, Tensor out)
      : MulScalar{Checked(input, alpha, out), alpha, out} {
    std::size_t bytes = 0;
    ContiguousBytes(input, &bytes);
    count_ = bytes / Width(dtype_);
  }

  bool capture_safe() const override { return false; }

  void operator()(const Tensor input, const double alpha,
                  Tensor out) const override {
    if (input.shape() != input_shape_ || out.shape() != out_shape_ ||
        input.strides() != input_strides_ || out.strides() != out_strides_ ||
        input.dtype() != dtype_ || out.dtype() != dtype_ ||
        input.device() != device_ || out.device() != device_ ||
        std::memcmp(&alpha, &alpha_, sizeof(alpha)))
      throw std::invalid_argument(
          "CPU MulScalar: call metadata or alpha changed after Make");
    if (!supports(input, alpha, out))
      throw std::invalid_argument(
          "CPU MulScalar: call pointer, layout or alias changed after Make");
    if (count_ == 0) return;
    // CPU scalar arithmetic retains float opmath precision even for half/BF16.
    const float multiplier = FloatAlpha(alpha_);
    DispatchFunc<Device::Type::kCpu,
                 List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        dtype_, [&](auto tag) {
          using T = typename decltype(tag)::type;
          const auto* src = static_cast<const T*>(input.data());
          auto* dst = static_cast<T*>(out.data());
#pragma omp parallel for
          for (std::size_t i = 0; i < count_; ++i) {
            const float product = Cast<float>(src[i]) * multiplier;
            if constexpr (IsBFloat16<Device::Type::kCpu, T>) {
              // Match CPU scalar BF16 conversion, not the shared caster's
              // sign/payload-preserving NaN conversion.
              dst[i] = std::isnan(product) ? T{std::uint16_t{0x7FC0}}
                                           : Cast<T>(product);
            } else {
              dst[i] = Cast<T>(product);
            }
          }
        }, "CPU MulScalar");
  }

 private:
  static bool SupportedDtype(DataType type) {
    return type == DataType::kFloat32 || type == DataType::kFloat16 ||
           type == DataType::kBFloat16;
  }

  static std::size_t Width(DataType type) {
    return type == DataType::kFloat32 ? 4u : 2u;
  }

  static bool ContiguousBytes(const Tensor t, std::size_t* bytes) {
    if (t.shape().size() != t.strides().size()) return false;
    const auto limit = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max()) / Width(t.dtype());
    bool empty = false;
    for (std::size_t axis = 0; axis < t.ndim(); ++axis) {
      if (t.stride(axis) < 0 || t.size(axis) > limit) return false;
      empty = empty || t.size(axis) == 0;
    }
    if (empty) { *bytes = 0; return true; }
    std::size_t count = 1;
    for (std::size_t axis = t.ndim(); axis > 0; --axis) {
      const auto size = t.size(axis - 1);
      if ((size > 1 && t.stride(axis - 1) !=
                           static_cast<std::ptrdiff_t>(count)) ||
          size > limit / count)
        return false;
      count *= size;
    }
    *bytes = count * Width(t.dtype());
    return true;
  }

  static bool ValidAddress(const Tensor t, std::size_t bytes) {
    if (bytes == 0) return true;
    const auto start = reinterpret_cast<std::uintptr_t>(t.data());
    return start && start % Width(t.dtype()) == 0 &&
        start <= std::numeric_limits<std::uintptr_t>::max() - bytes;
  }

  static float FloatAlpha(double alpha) {
    if (std::isfinite(alpha) &&
        std::fabs(alpha) > std::numeric_limits<float>::max())
      return std::copysign(std::numeric_limits<float>::infinity(), alpha);
    return static_cast<float>(alpha);
  }

  static Tensor Checked(const Tensor input, const double alpha, Tensor out) {
    if (!supports(input, alpha, out))
      throw std::invalid_argument(
          "CPU MulScalar: unsupported dtype/device/layout or alias");
    return input;
  }

  std::size_t count_ = 0;
};

}  // namespace infini::ops
#endif  // INFINI_OPS_CPU_MUL_SCALAR_MUL_SCALAR_H_
