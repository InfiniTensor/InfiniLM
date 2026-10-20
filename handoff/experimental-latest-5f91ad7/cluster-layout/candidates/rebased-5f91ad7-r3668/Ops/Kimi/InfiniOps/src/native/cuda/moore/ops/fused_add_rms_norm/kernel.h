#ifndef INFINI_OPS_MOORE_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_MOORE_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/fused_add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<FusedAddRmsNorm, Device::Type::kMoore>
    : public CudaFusedAddRmsNorm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaFusedAddRmsNorm<Runtime<Device::Type::kMoore>>::CudaFusedAddRmsNorm;
};

}  // namespace infini::ops

#endif
