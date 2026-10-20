#ifndef INFINI_CCL_DEVICES_CPU_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_CPU_DATA_TYPE_H_

#include "data_type_impl.h"
#include "devices/cpu/device_.h"

namespace infini::ccl {

struct Float16 {
  std::uint16_t bits;

  static inline Float16 FromFloat(float val) {
    std::uint32_t f32;
    std::memcpy(&f32, &val, sizeof(f32));
    std::uint16_t sign = (f32 >> 16) & 0x8000;
    std::int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
    std::uint32_t mantissa = f32 & 0x7FFFFF;

    if (exponent >= 16) {
      // NaN
      if (exponent == 128 && mantissa != 0) {
        return {static_cast<std::uint16_t>(sign | 0x7E00)};
      }
      // Inf
      return {static_cast<std::uint16_t>(sign | 0x7C00)};
    } else if (exponent >= -14) {
      return {static_cast<std::uint16_t>(sign | ((exponent + 15) << 10) |
                                         (mantissa >> 13))};
    } else if (exponent >= -24) {
      mantissa |= 0x800000;
      mantissa >>= (-14 - exponent);
      return {static_cast<std::uint16_t>(sign | (mantissa >> 13))};
    }
    // Too small for subnormal: return signed zero.
    return {sign};
  }

  inline float ToFloat() const {
    std::uint32_t sign = (bits & 0x8000) << 16;
    std::int32_t exponent = (bits >> 10) & 0x1F;
    std::uint32_t mantissa = bits & 0x3FF;
    std::uint32_t f32_bits;

    if (exponent == 31) {
      f32_bits = sign | 0x7F800000 | (mantissa << 13);
    } else if (exponent == 0) {
      if (mantissa == 0) {
        f32_bits = sign;
      } else {
        exponent = -14;
        while ((mantissa & 0x400) == 0) {
          mantissa <<= 1;
          exponent--;
        }
        mantissa &= 0x3FF;
        f32_bits = sign | ((exponent + 127) << 23) | (mantissa << 13);
      }
    } else {
      f32_bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &f32_bits, sizeof(result));
    return result;
  }
};

struct BFloat16 {
  std::uint16_t bits;

  static inline BFloat16 FromFloat(float val) {
    std::uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));

    const std::uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    std::uint16_t bf16_bits =
        static_cast<std::uint16_t>((bits32 + rounding_bias) >> 16);
    return {bf16_bits};
  }

  inline float ToFloat() const {
    std::uint32_t bits32 = static_cast<std::uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &bits32, sizeof(result));
    return result;
  }
};

template <>
struct TypeMap<Device::Type::kCpu, DataType::kFloat16> {
  using type = Float16;
};

template <>
struct TypeMap<Device::Type::kCpu, DataType::kBFloat16> {
  using type = BFloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CPU_DATA_TYPE_H_
