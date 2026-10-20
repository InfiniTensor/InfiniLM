#ifndef INFINI_OPS_MOORE_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_MOORE_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kMoore>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kMoore>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kMoore>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
