#ifndef INFINI_OPS_THEAD_ADD_KERNEL_H_
#define INFINI_OPS_THEAD_ADD_KERNEL_H_

#include "native/cuda/ops/add/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kThead>
    : public CudaAdd<Runtime<Device::Type::kThead>> {
 public:
  using CudaAdd<Runtime<Device::Type::kThead>>::CudaAdd;
};

}  // namespace infini::ops

#endif
