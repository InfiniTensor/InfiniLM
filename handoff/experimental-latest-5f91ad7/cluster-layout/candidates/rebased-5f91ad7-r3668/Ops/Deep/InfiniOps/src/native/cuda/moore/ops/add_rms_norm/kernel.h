#ifndef INFINI_OPS_MOORE_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_MOORE_ADD_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kMoore>
    : public CudaAddRmsNorm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kMoore>>::CudaAddRmsNorm;
};

}  // namespace infini::ops

#endif
