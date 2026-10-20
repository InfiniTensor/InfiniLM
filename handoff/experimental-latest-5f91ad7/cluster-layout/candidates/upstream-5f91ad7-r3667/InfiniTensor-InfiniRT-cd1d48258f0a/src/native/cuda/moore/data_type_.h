#ifndef INFINI_RT_MOORE_DATA_TYPE__H_
#define INFINI_RT_MOORE_DATA_TYPE__H_

#include <musa_bf16.h>
#include <musa_fp16.h>

#include "data_type.h"
#include "native/cuda/moore/device_.h"

namespace infini::rt {

using cuda_bfloat16 = __mt_bfloat16;

using cuda_bfloat162 = __mt_bfloat162;

template <>
struct TypeMap<Device::Type::kMoore, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kMoore, DataType::kBFloat16> {
  using type = __mt_bfloat16;
};

}  // namespace infini::rt

#endif
