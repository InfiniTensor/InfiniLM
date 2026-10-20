#ifndef INFINI_CCL_DATA_TYPE_IMPL_H_
#define INFINI_CCL_DATA_TYPE_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "constexpr_map.h"
#include "data_type.h"

namespace infini::ccl {

enum class DataType : int8_t {
  kChar = infinicclChar,
  kInt8 = infinicclInt8,
  kInt16 = infinicclInt16,
  kInt32 = infinicclInt32,
  kInt64 = infinicclInt64,
  kUInt8 = infinicclUInt8,
  kUInt16 = infinicclUInt16,
  kUInt32 = infinicclUInt32,
  kUInt64 = infinicclUInt64,
  kFloat16 = infinicclFloat16,
  kBFloat16 = infinicclBFloat16,
  kFloat32 = infinicclFloat32,
  kFloat64 = infinicclFloat64,
  kNumTypes = infinicclNumTypes,
};

constexpr ConstexprMap<DataType, std::size_t, 12> kDataTypeToSize{{{
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
}}};

constexpr ConstexprMap<DataType, std::string_view, 12> kDataTypeToDesc{{{
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
}}};

constexpr ConstexprMap<std::string_view, DataType, 12> kStringToDataType{{{
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
}}};

// `TypeMap` maps a `DataType` on a given `Device::Type` to the corresponding
// C++ type.
template <Device::Type dev, DataType dtype>
struct TypeMap;

template <Device::Type dev, DataType dtype>
using TypeMapType = typename TypeMap<dev, dtype>::type;

// `DataTypeMap` maps a C++ type to the corresponding `DataType`.
template <typename T>
struct DataTypeMap;

template <typename T>
inline constexpr DataType DataTypeMapValue = DataTypeMap<T>::value;

#define DEFINE_DATA_TYPE_MAPPING(ENUM_VALUE, CPP_TYPE)      \
  template <Device::Type dev>                               \
  struct TypeMap<dev, DataType::ENUM_VALUE> {               \
    using type = CPP_TYPE;                                  \
  };                                                        \
                                                            \
  template <>                                               \
  struct DataTypeMap<CPP_TYPE> {                            \
    static constexpr DataType value = DataType::ENUM_VALUE; \
  };

DEFINE_DATA_TYPE_MAPPING(kUInt8, std::uint8_t)
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

}  // namespace infini::ccl

#endif  // INFINI_CCL_DATA_TYPE_IMPL_H_
