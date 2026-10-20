#ifndef INFINI_OPS_METAX_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_METAX_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/fused_add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kMetax>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaFusedAddRmsNorm<Runtime<Device::Type::kMetax>>::CudaFusedAddRmsNorm;
};

}  // namespace infini::ops

#endif
