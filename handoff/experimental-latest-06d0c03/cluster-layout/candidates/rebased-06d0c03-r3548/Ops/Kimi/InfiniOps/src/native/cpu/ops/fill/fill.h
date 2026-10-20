#ifndef INFINI_OPS_CPU_FILL_FILL_H_
#define INFINI_OPS_CPU_FILL_FILL_H_

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "base/fill.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

// Native scalar Fill provider. Tensor-valued Fill remains unsupported.
template <>
class Operator<Fill, Device::Type::kCpu, 0> : public Fill,
                                              Caster<Device::Type::kCpu> {
 public:
  static bool supports(const Tensor input, const double value, Tensor out) {
    return input.device().type() == Device::Type::kCpu &&
           Fill::supports(input, value, out) && valid_output(out);
  }

  static bool supports(const Tensor, const Tensor, Tensor) { return false; }

  Operator(const Tensor input, const double value, Tensor out)
      : Fill{Checked(input, value, out), value, out} {}

  Operator(const Tensor input, const Tensor value, Tensor out)
      : Fill{input, value, out} {
    throw std::invalid_argument(
        "CPU Fill does not supply a Tensor-valued provider");
  }

  void operator()(const Tensor input, const double value,
                  Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        input.dtype() != input_type_ || out.shape() != out_shape_ ||
        out.strides() != out_strides_ || out.dtype() != out_type_ ||
        out.device().index() != device_index_ ||
        !(value == value_ || (std::isnan(value) && std::isnan(value_))) ||
        !supports(input, value, out))
      throw std::invalid_argument("CPU Fill metadata/value changed after Make");
    if (out.numel() == 0) return;
    DispatchFunc<Device::Type::kCpu, AllTypes>(
        out.dtype(),
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          const T scalar = Cast<T>(value);
          auto* data = static_cast<std::byte*>(out.data());
          // Contiguous fastpath is conditional on the actual output layout.
          if (out.IsContiguous()) {
            for (size_t index = 0; index < out.numel(); ++index)
              std::memcpy(data + index * sizeof(T), &scalar, sizeof(T));
            return;
          }
          for (size_t linear = 0; linear < out.numel(); ++linear) {
            size_t remaining = linear, offset = 0;
            for (size_t dim = out.ndim(); dim-- > 0;) {
              offset += (remaining % out.size(dim)) *
                        static_cast<size_t>(out.stride(dim));
              remaining /= out.size(dim);
            }
            std::memcpy(data + offset * sizeof(T), &scalar, sizeof(T));
          }
        },
        "CPU scalar Fill");
  }

  void operator()(const Tensor, const Tensor, Tensor) const override {
    throw std::invalid_argument(
        "CPU Fill does not supply a Tensor-valued provider");
  }

 private:
  static Tensor Checked(const Tensor input, double value, Tensor out) {
    if (!supports(input, value, out))
      throw std::invalid_argument(
          "CPU Fill requires valid scalar fill metadata");
    return input;
  }
};
}  // namespace infini::ops
#endif
