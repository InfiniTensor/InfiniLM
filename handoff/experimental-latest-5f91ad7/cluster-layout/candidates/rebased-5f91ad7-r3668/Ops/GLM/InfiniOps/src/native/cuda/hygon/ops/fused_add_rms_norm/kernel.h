#ifndef INFINI_OPS_HYGON_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_HYGON_FUSED_ADD_RMS_NORM_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/fused_add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kHygon>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kHygon>> {
 public:
  using CudaFusedAddRmsNorm<Runtime<Device::Type::kHygon>>::CudaFusedAddRmsNorm;
};

}  // namespace infini::ops

#endif
