#ifndef INFINI_OPS_MARS_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_MARS_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kMars>
    : public CudaAddRmsNorm<Runtime<Device::Type::kMars>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kMars>>::CudaAddRmsNorm;
};

}  // namespace infini::ops

#endif
