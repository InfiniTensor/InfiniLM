#ifndef INFINI_OPS_THEAD_SWIGLU_KERNEL_H_
#define INFINI_OPS_THEAD_SWIGLU_KERNEL_H_

#include "native/cuda/ops/swiglu/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kThead>
    : public CudaSwiglu<Runtime<Device::Type::kThead>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kThead>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
