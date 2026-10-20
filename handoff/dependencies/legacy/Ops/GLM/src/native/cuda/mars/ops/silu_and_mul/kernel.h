#ifndef INFINI_OPS_MARS_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_MARS_SILU_AND_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kMars>
    : public CudaSiluAndMul<Runtime<Device::Type::kMars>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kMars>>::CudaSiluAndMul;
};

}  // namespace infini::ops

#endif
