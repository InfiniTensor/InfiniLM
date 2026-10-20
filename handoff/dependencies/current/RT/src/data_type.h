#ifndef INFINI_RT_DATA_TYPE_H_
#define INFINI_RT_DATA_TYPE_H_

#include <cstdint>
#include <cstring>
#include <string>

#include "common/constexpr_map.h"
#include "common/traits.h"
#include "device.h"

namespace infini::rt {

enum class DataType : std::int8_t {
  kInt8,
  kInt16,
  kInt32,
  kInt64,
  kUInt8,
  kUInt16,
  kUInt32,
  kUInt64,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
  kBool
};

constexpr ConstexprMap<DataType, std::size_t, 13> kDataTypeToSize{{{
    {DataType::kInt8, 1},
    {DataType::kInt16, 2},
    {DataType::kInt32, 4},
    {DataType::kInt64, 8},
    {DataType::kUInt8, 1},
    {DataType::kUInt16, 2},
    {DataType::kUInt32, 4},
    {DataType::kUInt64, 8},
    {DataType::kFloat16, 2},
    {DataType::kBFloat16, 2},
    {DataType::kFloat32, 4},
    {DataType::kFloat64, 8},
    {DataType::kBool, 1},
}}};

constexpr ConstexprMap<DataType, std::string_view, 13> kDataTypeToDesc{{{
    {DataType::kInt8, "int8"},
    {DataType::kInt16, "int16"},
    {DataType::kInt32, "int32"},
    {DataType::kInt64, "int64"},
    {DataType::kUInt8, "uint8"},
    {DataType::kUInt16, "uint16"},
    {DataType::kUInt32, "uint32"},
    {DataType::kUInt64, "uint64"},
    {DataType::kFloat16, "float16"},
    {DataType::kBFloat16, "bfloat16"},
    {DataType::kFloat32, "float32"},
    {DataType::kFloat64, "float64"},
    {DataType::kBool, "bool"},
}}};

constexpr ConstexprMap<std::string_view, DataType, 13> kStringToDataType{{{
    {"int8", DataType::kInt8},
    {"int16", DataType::kInt16},
    {"int32", DataType::kInt32},
    {"int64", DataType::kInt64},
    {"uint8", DataType::kUInt8},
    {"uint16", DataType::kUInt16},
    {"uint32", DataType::kUInt32},
    {"uint64", DataType::kUInt64},
    {"float16", DataType::kFloat16},
    {"bfloat16", DataType::kBFloat16},
    {"float32", DataType::kFloat32},
    {"float64", DataType::kFloat64},
    {"bool", DataType::kBool},
}}};

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
      mantissa += 0x0FFF + ((mantissa >> 13) & 1);
      return {static_cast<std::uint16_t>(
          sign | (((exponent + 15) << 10) + (mantissa >> 13)))};
    } else if (exponent >= -25) {
      mantissa |= 0x800000;
      const unsigned shift = static_cast<unsigned>(-exponent - 1);
      const auto bias = ((1u << (shift - 1)) - 1) + ((mantissa >> shift) & 1);
      return {static_cast<std::uint16_t>(sign | ((mantissa + bias) >> shift))};
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

template <Device::Type dev, DataType dtype>
struct TypeMap;

template <Device::Type dev, DataType dtype>
using TypeMapType = typename TypeMap<dev, dtype>::type;

#define DEFINE_DATA_TYPE_MAPPING(ENUM_VALUE, CPP_TYPE) \
  template <Device::Type dev>                          \
  struct TypeMap<dev, DataType::ENUM_VALUE> {          \
    using type = CPP_TYPE;                             \
  };

DEFINE_DATA_TYPE_MAPPING(kUInt8, std::uint8_t)
DEFINE_DATA_TYPE_MAPPING(kBool, bool)
DEFINE_DATA_TYPE_MAPPING(kInt8, std::int8_t)
DEFINE_DATA_TYPE_MAPPING(kUInt16, std::uint16_t)
DEFINE_DATA_TYPE_MAPPING(kInt16, std::int16_t)
DEFINE_DATA_TYPE_MAPPING(kUInt32, std::uint32_t)
DEFINE_DATA_TYPE_MAPPING(kInt32, std::int32_t)
DEFINE_DATA_TYPE_MAPPING(kUInt64, std::uint64_t)
DEFINE_DATA_TYPE_MAPPING(kInt64, std::int64_t)
DEFINE_DATA_TYPE_MAPPING(kFloat32, float)
DEFINE_DATA_TYPE_MAPPING(kFloat64, double)
#undef DEFINE_DATA_TYPE_MAPPING

// Checks whether a C++ type is the bfloat16 or float16 type for the given
// device. Full specializations for each device's float16/bfloat16 types are
// provided in the corresponding platform-specific device type headers.
template <Device::Type dev, typename T>
inline constexpr bool IsBFloat16 =
    std::is_same_v<T, TypeMapType<dev, DataType::kBFloat16>>;

template <Device::Type dev, typename T>
inline constexpr bool IsFP16 =
    std::is_same_v<T, TypeMapType<dev, DataType::kFloat16>>;

// Defines the common categories of data types using List.
using FloatTypes = List<DataType::kFloat32, DataType::kFloat64>;
using ReducedFloatTypes = List<DataType::kFloat16, DataType::kBFloat16>;
using IntTypes =
    List<DataType::kInt8, DataType::kInt16, DataType::kInt32, DataType::kInt64>;
using UIntTypes = List<DataType::kUInt8, DataType::kUInt16, DataType::kUInt32,
                       DataType::kUInt64>;

using BitTypes8 = List<DataType::kInt8, DataType::kUInt8>;
using BitTypes16 = List<DataType::kInt16, DataType::kUInt16, DataType::kFloat16,
                        DataType::kBFloat16>;
using BitTypes32 =
    List<DataType::kInt32, DataType::kUInt32, DataType::kFloat32>;
using BitTypes64 =
    List<DataType::kInt64, DataType::kUInt64, DataType::kFloat64>;

using AllFloatTypes = ConcatType<FloatTypes, ReducedFloatTypes>;
using AllIntTypes = ConcatType<IntTypes, UIntTypes>;
using AllTypes = ConcatType<AllFloatTypes, AllIntTypes>;

}  // namespace infini::rt

#endif
