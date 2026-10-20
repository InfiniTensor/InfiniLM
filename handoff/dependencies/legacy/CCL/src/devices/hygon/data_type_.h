#ifndef INFINI_CCL_DEVICES_HYGON_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_HYGON_DATA_TYPE_H_

// clang-format off
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
// clang-format on

#include "data_type_impl.h"
#include "devices/hygon/device_.h"

namespace infini::ccl {

template <>
struct TypeMap<Device::Type::kHygon, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kHygon, DataType::kBFloat16> {
  using type = hip_bfloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_HYGON_DATA_TYPE_H_
