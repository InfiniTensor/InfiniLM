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
#include "native/cpu/dense_view.h"

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
    return cpu_dense_view::DisjointOrSameContiguous(input, out);
  }

  Operator(const Tensor input, const double eps, Tensor out)
      : RowL2Normalize{Checked(input, eps, out), eps, out} {
    cpu_dense_view::ContiguousBytes(input, &bytes_);
  }

  void operator()(const Tensor input, const double eps,
                  Tensor out) const override {
    if (!std::equal_to<Tensor>{}(input, input_meta_) ||
        !std::equal_to<Tensor>{}(out, out_meta_) ||
        std::memcmp(&eps, &eps_, sizeof(eps)))
      throw std::invalid_argument(
          "CPU RowL2Normalize: call metadata or eps changed after Make");
    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    if (!cpu_dense_view::ValidAddress(input, bytes_) ||
        !cpu_dense_view::ValidAddress(out, bytes_) ||
        (bytes_ && ip < op + bytes_ && op < ip + bytes_ &&
         !(ip == op && input_meta_.strides() == out_meta_.strides())))
      throw std::invalid_argument(
          "CPU RowL2Normalize: call pointer or alias changed after "
          "Make");
    if (input.size(0) == 0 || input.size(1) == 0) return;
    const float epsilon = eps_ > std::numeric_limits<float>::max()
                              ? std::numeric_limits<float>::infinity()
                              : static_cast<float>(eps_);
    DispatchFunc<
        Device::Type::kCpu,
        List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        input_meta_.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          const auto* src = static_cast<const T*>(input.data());
          auto* dst = static_cast<T*>(out.data());
          const auto rows = input.size(0), cols = input.size(1);
          const float eps_t = Cast<float>(Round<T>(epsilon));
#pragma omp parallel for
          for (std::size_t row = 0; row < rows; ++row) {
            const auto offset = row * cols;
            const float norm = Cast<float>(
                Round<T>(std::sqrt(SumSquares(src + offset, cols))));
            // Preserve NaNs and the norm's T rounding before division.
            const float denominator =
                std::isnan(norm) ? norm : (norm < eps_t ? eps_t : norm);
            for (std::size_t col = 0; col < cols; ++col) {
              const float value = Cast<float>(src[offset + col]);
              dst[offset + col] = Round<T>(value / denominator);
            }
          }
        },
        "CPU RowL2Normalize");
  }

 private:
  std::size_t bytes_ = 0;
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

  static Tensor Checked(const Tensor input, const double eps, Tensor out) {
    if (!supports(input, eps, out))
      throw std::invalid_argument(
          "CPU RowL2Normalize: unsupported dtype/device/layout, eps or alias");
    return input;
  }
};

}  // namespace infini::ops
#endif  // INFINI_OPS_CPU_ROW_L2_NORMALIZE_H_
