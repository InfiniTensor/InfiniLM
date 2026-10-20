#ifndef INFINI_OPS_DATA_TYPE_H_
#define INFINI_OPS_DATA_TYPE_H_

#include <infini/rt.h>

#include "common/traits.h"
#include "device.h"

namespace infini::ops {

using infini::rt::DataType;

using infini::rt::BFloat16;
using infini::rt::Float16;

using infini::rt::kDataTypeToDesc;
using infini::rt::kDataTypeToSize;
using infini::rt::kStringToDataType;

template <Device::Type dev, DataType dtype>
using TypeMap = infini::rt::TypeMap<dev, dtype>;

template <Device::Type dev, DataType dtype>
using TypeMapType = infini::rt::TypeMapType<dev, dtype>;

template <Device::Type dev, typename T>
inline constexpr bool IsBFloat16 = infini::rt::IsBFloat16<dev, T>;

template <Device::Type dev, typename T>
inline constexpr bool IsFP16 = infini::rt::IsFP16<dev, T>;

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

}  // namespace infini::ops

#endif
