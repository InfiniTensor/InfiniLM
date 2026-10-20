#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_

#include "backends/ccl/nccl/api.h"
#include "devices/nvidia/runtime_.h"

namespace infini::ccl {

template <>
struct NcclDataTypeTraits<Device::Type::kNvidia> {
#if defined(__CUDA_BF16_TYPES_EXIST__)
  static constexpr ncclDataType_t kBFloat16 = ncclBfloat16;
#else
  static constexpr ncclDataType_t kBFloat16 = ncclNumTypes;
#endif
};

template <>
struct CclApi<BackendType::kNccl, Device::Type::kNvidia>
    : NcclApi<Device::Type::kNvidia> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_
