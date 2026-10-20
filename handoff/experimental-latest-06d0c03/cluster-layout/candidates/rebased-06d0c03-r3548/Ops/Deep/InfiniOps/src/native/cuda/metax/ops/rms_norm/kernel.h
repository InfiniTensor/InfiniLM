#ifndef INFINI_OPS_METAX_RMS_NORM_KERNEL_H_
#define INFINI_OPS_METAX_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kMetax>
    : public CudaRmsNorm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaRmsNorm<Runtime<Device::Type::kMetax>>::CudaRmsNorm;
};

}  // namespace infini::ops

#endif
