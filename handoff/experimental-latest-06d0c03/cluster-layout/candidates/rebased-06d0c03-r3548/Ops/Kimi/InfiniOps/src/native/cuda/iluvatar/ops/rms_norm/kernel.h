#ifndef INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kIluvatar>
    : public CudaRmsNorm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kIluvatar>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
