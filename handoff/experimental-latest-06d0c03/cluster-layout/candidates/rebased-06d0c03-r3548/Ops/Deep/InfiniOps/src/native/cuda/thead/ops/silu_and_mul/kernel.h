#ifndef INFINI_OPS_THEAD_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_THEAD_SILU_AND_MUL_KERNEL_H_

#include "native/cuda/ops/silu_and_mul/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kThead>
    : public CudaSiluAndMul<Runtime<Device::Type::kThead>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kThead>>::CudaSiluAndMul;
};

}  // namespace infini::ops

#endif
