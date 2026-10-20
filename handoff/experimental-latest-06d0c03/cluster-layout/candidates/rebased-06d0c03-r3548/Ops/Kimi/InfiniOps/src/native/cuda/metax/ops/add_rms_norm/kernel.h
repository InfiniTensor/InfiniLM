#ifndef INFINI_OPS_METAX_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_METAX_ADD_RMS_NORM_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/add_rms_norm/kernel.h"

namespace infini::ops {

template <>
class Operator<AddRmsNorm, Device::Type::kMetax>
    : public CudaAddRmsNorm<Runtime<Device::Type::kMetax>> {
 public:
  using CudaAddRmsNorm<Runtime<Device::Type::kMetax>>::CudaAddRmsNorm;
};

}  // namespace infini::ops

#endif
