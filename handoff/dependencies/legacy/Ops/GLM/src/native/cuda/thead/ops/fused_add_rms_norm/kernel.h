#ifndef INFINI_OPS_THEAD_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_THEAD_FUSED_ADD_RMS_NORM_KERNEL_H_

#include "native/cuda/ops/fused_add_rms_norm/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kThead>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kThead>> {
 public:
  using CudaFusedAddRmsNorm<Runtime<Device::Type::kThead>>::CudaFusedAddRmsNorm;
};

}  // namespace infini::ops

#endif
