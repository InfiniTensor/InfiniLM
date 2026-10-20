#ifndef INFINI_OPS_CPU_MUL_SCALAR_H_
#define INFINI_OPS_CPU_MUL_SCALAR_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>

#include "base/mul.h"
#include "native/cpu/caster_.h"
#include "native/cpu/dense_view.h"

namespace infini::ops {

class CpuScalarMul : Caster<Device::Type::kCpu> {
 public:
  static bool supports(const Tensor input, const double, Tensor out) {
    if (input.device().type() != Device::Type::kCpu ||
        input.device() != out.device() || input.dtype() != out.dtype() ||
        input.shape() != out.shape() || !SupportedDtype(input.dtype()))
      return false;
    return cpu_dense_view::DisjointOrSameContiguous(input, out);
  }

  CpuScalarMul(const Tensor input, const double alpha, Tensor out)
      : input_meta_(Checked(input, alpha, out)), out_meta_(out), alpha_(alpha) {
    std::size_t bytes = 0;
    cpu_dense_view::ContiguousBytes(input, &bytes);
    count_ = bytes / input.element_size();
  }

  void Run(const Tensor input, const double alpha, Tensor out) const {
    if (!std::equal_to<Tensor>{}(input, input_meta_) ||
        !std::equal_to<Tensor>{}(out, out_meta_) ||
        std::memcmp(&alpha, &alpha_, sizeof(alpha)))
      throw std::invalid_argument(
          "CPU scalar Mul: call metadata or alpha changed after Make");
    const auto bytes = count_ * input_meta_.element_size();
    const auto ip = reinterpret_cast<std::uintptr_t>(input.data());
    const auto op = reinterpret_cast<std::uintptr_t>(out.data());
    if (!cpu_dense_view::ValidAddress(input, bytes) ||
        !cpu_dense_view::ValidAddress(out, bytes) ||
        (bytes && ip < op + bytes && op < ip + bytes &&
         !(ip == op && input_meta_.strides() == out_meta_.strides())))
      throw std::invalid_argument(
          "CPU scalar Mul: call pointer or alias changed after Make");
    if (count_ == 0) return;
    // CPU scalar arithmetic retains float opmath precision even for half/BF16.
    const float multiplier = FloatAlpha(alpha_);
    DispatchFunc<
        Device::Type::kCpu,
        List<DataType::kFloat32, DataType::kFloat16, DataType::kBFloat16>>(
        input_meta_.dtype(),
        [&](auto tag) {
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
        },
        "CPU scalar Mul");
  }

 private:
  static bool SupportedDtype(DataType type) {
    return type == DataType::kFloat32 || type == DataType::kFloat16 ||
           type == DataType::kBFloat16;
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
          "CPU scalar Mul: unsupported dtype/device/layout or alias");
    return input;
  }

  std::size_t count_ = 0;
  const Tensor input_meta_, out_meta_;
  double alpha_;
};

}  // namespace infini::ops
#endif  // INFINI_OPS_CPU_MUL_SCALAR_H_
