#ifndef INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_NVIDIA_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kNvidia>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kNvidia>>::CudaReshapeAndCacheFlash;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
