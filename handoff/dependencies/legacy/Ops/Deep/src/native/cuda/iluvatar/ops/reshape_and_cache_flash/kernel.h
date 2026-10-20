#ifndef INFINI_OPS_ILUVATAR_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/reshape_and_cache_flash/kernel.h"

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kIluvatar>
    : public CudaReshapeAndCacheFlash<Runtime<Device::Type::kIluvatar>, 128> {
 public:
  // BI-V150 reports a 2048-thread device limit, but CoreX faults when this
  // kernel uses the resulting 1024/2048-thread launch. The operator supports
  // head dimensions up to 128 here, so 128 threads cover every element while
  // keeping the workaround local to the Iluvatar provider.
  using CudaReshapeAndCacheFlash<Runtime<Device::Type::kIluvatar>,
                                 128>::CudaReshapeAndCacheFlash;
};

}  // namespace infini::ops

#endif
