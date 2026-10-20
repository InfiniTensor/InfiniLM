#ifndef INFINI_OPS_MARS_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_MARS_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/fused_add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kMars>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kMars>> {
 public:
  using CudaFusedAddRmsNorm<Runtime<Device::Type::kMars>>::CudaFusedAddRmsNorm;
};

}  // namespace infini::ops

#endif
