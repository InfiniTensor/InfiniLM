#ifndef INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_
#define INFINI_OPS_HYGON_RMS_NORM_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kHygon>
    : public CudaRmsNorm<Runtime<Device::Type::kHygon>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kHygon>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
