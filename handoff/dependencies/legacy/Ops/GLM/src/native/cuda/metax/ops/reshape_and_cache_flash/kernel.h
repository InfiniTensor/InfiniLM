#ifndef INFINI_OPS_METAX_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_METAX_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kMetax>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kMetax>> {
 public:
  using CudaReshapeAndCacheFlash<
      Runtime<Device::Type::kMetax>>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
