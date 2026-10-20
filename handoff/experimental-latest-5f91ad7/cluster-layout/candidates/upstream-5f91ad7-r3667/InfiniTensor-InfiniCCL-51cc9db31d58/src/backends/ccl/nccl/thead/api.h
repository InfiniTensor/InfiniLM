#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_THEAD_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_THEAD_API_H_

#include "backends/ccl/nccl/api.h"
#include "devices/thead/runtime_.h"

namespace infini::ccl {

template <>
struct NcclDataTypeTraits<Device::Type::kThead> {
#if defined(__CUDA_BF16_TYPES_EXIST__)
  static constexpr ncclDataType_t kBFloat16 = ncclBfloat16;
#else
  static constexpr ncclDataType_t kBFloat16 = ncclNumTypes;
#endif
};

template <>
struct CclApi<BackendType::kNccl, Device::Type::kThead>
    : NcclApi<Device::Type::kThead> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_THEAD_API_H_
