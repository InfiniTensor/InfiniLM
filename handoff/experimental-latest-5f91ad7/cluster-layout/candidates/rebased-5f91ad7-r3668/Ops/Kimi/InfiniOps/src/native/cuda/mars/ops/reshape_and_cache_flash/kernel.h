#ifndef INFINI_OPS_MARS_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_MARS_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kMars>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kMars>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kMars>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
