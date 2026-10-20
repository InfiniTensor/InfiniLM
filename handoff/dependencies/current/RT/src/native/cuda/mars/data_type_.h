#ifndef INFINI_RT_MARS_DATA_TYPE__H_
#define INFINI_RT_MARS_DATA_TYPE__H_

#include <common/hpcc_bfloat16.h>
#include <common/hpcc_fp16.h>
#include <hcr/hc_runtime.h>

#include "data_type.h"
#include "native/cuda/mars/device_.h"

namespace infini::rt {

using cuda_bfloat16 = __hpcc_bfloat16;

using cuda_bfloat162 = __hpcc_bfloat162;

template <>
struct TypeMap<Device::Type::kMars, DataType::kFloat16> {
  using type = __half;
};

template <>
struct TypeMap<Device::Type::kMars, DataType::kBFloat16> {
  using type = __hpcc_bfloat16;
};

}  // namespace infini::rt

#endif
