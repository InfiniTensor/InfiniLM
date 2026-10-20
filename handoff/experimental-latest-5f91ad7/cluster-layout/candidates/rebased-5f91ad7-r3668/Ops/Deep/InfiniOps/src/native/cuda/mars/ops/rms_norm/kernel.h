#ifndef INFINI_OPS_MARS_RMS_NORM_KERNEL_H_
#define INFINI_OPS_MARS_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kMars>
    : public CudaRmsNorm<Runtime<Device::Type::kMars>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kMars>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
