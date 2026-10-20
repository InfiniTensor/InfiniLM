#ifndef INFINI_OPS_HYGON_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_HYGON_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kHygon>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kHygon>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kHygon>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
