#ifndef INFINI_RT_NVIDIA_DRIVER__H_
#define INFINI_RT_NVIDIA_DRIVER__H_

// clang-format off
#include <cuda.h>
// clang-format on

#include "native/cuda/driver_.h"
#include "native/cuda/nvidia/device_.h"

namespace infini::rt::driver {

template <>
struct Driver<Device::Type::kNvidia>
    : CudaDriver<Driver<Device::Type::kNvidia>> {
  using Result = CUresult;

  using Module = CUmodule;

  using Function = CUfunction;

  using Stream = CUstream;

  static constexpr Device::Type kDeviceType = Device::Type::kNvidia;

  static constexpr Result kSuccess = CUDA_SUCCESS;

  static constexpr auto ModuleLoadData = cuModuleLoadData;

  static constexpr auto ModuleGetFunction = cuModuleGetFunction;

  static constexpr auto ModuleUnload = cuModuleUnload;

  static constexpr auto FuncGetAttribute = cuFuncGetAttribute;

  static constexpr auto kFuncAttributeSharedSizeBytes =
      CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;

  static constexpr auto FuncSetCacheConfig = cuFuncSetCacheConfig;

  static constexpr auto kFuncCachePreferShared = CU_FUNC_CACHE_PREFER_SHARED;

  static constexpr auto FuncSetAttribute = cuFuncSetAttribute;

  static constexpr auto kFuncAttributeMaxDynamicSharedSizeBytes =
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;

  static constexpr auto LaunchKernel = cuLaunchKernel;
};

static_assert(Driver<Device::Type::kNvidia>::Validate());

}  // namespace infini::rt::driver

#endif
