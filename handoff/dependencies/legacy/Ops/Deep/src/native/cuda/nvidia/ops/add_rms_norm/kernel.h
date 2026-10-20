#ifndef INFINI_OPS_NVIDIA_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kNvidia>
    : public CudaAddRmsNorm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kNvidia>>::CudaAddRmsNorm;
};

}  // namespace infini::ops

#endif
