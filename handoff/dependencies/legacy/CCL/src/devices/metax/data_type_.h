#ifndef INFINI_CCL_DEVICES_METAX_DATA_TYPE_H_
#define INFINI_CCL_DEVICES_METAX_DATA_TYPE_H_

// clang-format off
#include <common/maca_bfloat16.h>
#include <common/maca_fp16.h>
#include <mcr/mc_runtime.h>
// clang-format on

#include "data_type_impl.h"
#include "devices/metax/device_.h"

namespace infini::ccl {

template <>
struct TypeMap<Device::Type::kMetax, DataType::kFloat16> {
  using type = __half;
};

template <>
struct TypeMap<Device::Type::kMetax, DataType::kBFloat16> {
  using type = __maca_bfloat16;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_METAX_DATA_TYPE_H_
