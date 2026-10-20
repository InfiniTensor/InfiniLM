#ifndef INFINI_OPS_THEAD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_THEAD_RMS_NORM_KERNEL_H_

#include "native/cuda/ops/rms_norm/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kThead>
    : public CudaRmsNorm<Runtime<Device::Type::kThead>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kThead>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
