#ifndef INFINI_OPS_ILUVATAR_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/fused_add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kIluvatar>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaFusedAddRmsNorm<
      Runtime<Device::Type::kIluvatar>>::CudaFusedAddRmsNorm;
};

}  // namespace infini::ops

#endif
