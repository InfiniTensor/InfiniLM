#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_HYGON_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_HYGON_API_H_

#include "backends/ccl/nccl/api.h"
#include "devices/hygon/runtime_.h"

namespace infini::ccl {

template <>
struct NcclDataTypeTraits<Device::Type::kHygon> {
#if defined(RCCL_BFLOAT16) && RCCL_BFLOAT16
  static constexpr ncclDataType_t kBFloat16 = ncclBfloat16;
#else
  static constexpr ncclDataType_t kBFloat16 = ncclNumTypes;
#endif
};

template <>
struct CclApi<BackendType::kNccl, Device::Type::kHygon>
    : NcclApi<Device::Type::kHygon> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_HYGON_API_H_
