#ifndef INFINI_OPS_MOORE_RMS_NORM_KERNEL_H_
#define INFINI_OPS_MOORE_RMS_NORM_KERNEL_H_

#include <utility>

// clang-format off
#include <musa_runtime.h>
// clang-format on

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kMoore>
    : public CudaRmsNorm<Runtime<Device::Type::kMoore>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kMoore>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
