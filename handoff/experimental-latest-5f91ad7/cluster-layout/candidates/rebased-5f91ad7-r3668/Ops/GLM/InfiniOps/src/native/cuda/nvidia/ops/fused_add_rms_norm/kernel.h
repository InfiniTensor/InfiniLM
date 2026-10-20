#ifndef INFINI_OPS_NVIDIA_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/fused_add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kNvidia>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaFusedAddRmsNorm<
      Runtime<Device::Type::kNvidia>>::CudaFusedAddRmsNorm;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
