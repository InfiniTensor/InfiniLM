#ifndef INFINI_OPS_ILUVATAR_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kIluvatar>
    : public CudaAddRmsNorm<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kIluvatar>>::CudaAddRmsNorm;
};

}  // namespace infini::ops

#endif
