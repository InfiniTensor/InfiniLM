#ifndef INFINI_OPS_HYGON_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_HYGON_SILU_AND_MUL_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kHygon>
    : public CudaSiluAndMul<Runtime<Device::Type::kHygon>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kHygon>>::CudaSiluAndMul;
};

}  // namespace infini::ops

#endif
