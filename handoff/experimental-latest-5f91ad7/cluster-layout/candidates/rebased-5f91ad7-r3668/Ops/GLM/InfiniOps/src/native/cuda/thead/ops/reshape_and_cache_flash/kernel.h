#ifndef INFINI_OPS_THEAD_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_THEAD_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <utility>

#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kThead>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kThead>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kThead>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
