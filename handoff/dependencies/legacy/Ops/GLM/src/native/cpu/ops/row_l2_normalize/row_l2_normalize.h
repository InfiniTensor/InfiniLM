#ifndef INFINI_OPS_CPU_ROW_L2_NORMALIZE_H_
#define INFINI_OPS_CPU_ROW_L2_NORMALIZE_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "base/row_l2_normalize.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

template <>
class Operator<RowL2Normalize, Device::Type::kCpu, 0>
    : public RowL2Normalize, Caster<Device::Type::kCpu> {
 public:
  static bool supports(const Tensor input, const double eps, Tensor out) {
    if (input.device().type() != Device::Type::kCpu ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        input.shape() != out.shape() || input.ndim() != 2 ||
        !SupportedDtype(input.dtype()) || !std::isfinite(eps) || eps < 0)
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

  Operator(const Tensor input, const double eps, Tensor out)
      : RowL2Normalize{Checked(input, eps, out), eps, out} {}

  bool capture_safe() const override { return false; }

  void operator()(const Tensor input, const double eps,
                  Tensor out) const override {
    if (input.shape() != input_shape_ || out.shape() != out_shape_ ||
        input.strides() != input_strides_ || out.strides() != out_strides_ ||
        input.dtype() != dtype_ || out.dtype() != dtype_ ||
        input.device() != device_ || out.device() != device_ ||
        std::memcmp(&eps, &eps_, sizeof(eps)))
      throw std::invalid_argument(
          "CPU RowL2Normalize: call metadata or eps changed after Make");
    if (!supports(input, eps, out))
      throw std::invalid_argument(
          "CPU RowL2Normalize: call pointer, layout or alias changed after Make");
    if (input.size(0) == 0 || input.size(1) == 0) return;
    const float epsilon = eps_ > std::numeric_limits<float>::max()
        ? std::numeric_limits<float>::infinity() : static_cast<float>(eps_);
    DispatchFunc<Device::Type::kCpu,
                 List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        dtype_, [&](auto tag) {
          using T = typename decltype(tag)::type;
          const auto* src = static_cast<const T*>(input.data());
          auto* dst = static_cast<T*>(out.data());
          const auto rows = input.size(0), cols = input.size(1);
          const float eps_t = Cast<float>(Round<T>(epsilon));
#pragma omp parallel for
          for (std::size_t row = 0; row < rows; ++row) {
            const auto offset = row * cols;
            const float norm = Cast<float>(Round<T>(
                std::sqrt(SumSquares(src + offset, cols))));
            // Preserve NaNs and the norm's T rounding before division.
            const float denominator = std::isnan(norm) ? norm
                : (norm < eps_t ? eps_t : norm);
            for (std::size_t col = 0; col < cols; ++col) {
              const float value = Cast<float>(src[offset + col]);
              dst[offset + col] = Round<T>(value / denominator);
            }
          }
        }, "CPU RowL2Normalize");
  }

 private:
  template <typename T>
  static T Round(float value) {
    if constexpr (IsBFloat16<Device::Type::kCpu, T>) {
      // CPU BF16 arithmetic canonicalizes NaNs to a positive quiet NaN.
      return std::isnan(value) ? T{std::uint16_t{0x7FC0}} : Cast<T>(value);
    } else {
      return Cast<T>(value);
    }
  }

  template <typename T>
  static float SumSquares(const T* src, std::size_t count) {
    if (count > 32) {
      const auto half = count / 2;
      const float left = SumSquares(src, half);
      const float right = SumSquares(src + half, count - half);
      return left + right;
    }
    // Bounded leaf storage, never a float copy of a row or checkpoint head.
    float partial[32] = {};
    for (std::size_t i = 0; i < count; ++i) {
      const float value = Cast<float>(src[i]);
      partial[i] = value * value;
    }
    for (std::size_t step = 1; step < 32; step *= 2)
      for (std::size_t i = 0; i < 32; i += 2 * step)
        partial[i] = partial[i] + partial[i + step];
    return partial[0];
  }

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

  static Tensor Checked(const Tensor input, const double eps, Tensor out) {
    if (!supports(input, eps, out))
      throw std::invalid_argument(
          "CPU RowL2Normalize: unsupported dtype/device/layout, eps or alias");
    return input;
  }
};

}  // namespace infini::ops
#endif  // INFINI_OPS_CPU_ROW_L2_NORMALIZE_H_
