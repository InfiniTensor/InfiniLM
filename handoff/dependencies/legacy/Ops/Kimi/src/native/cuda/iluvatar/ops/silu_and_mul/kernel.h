#ifndef INFINI_OPS_ILUVATAR_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SILU_AND_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kIluvatar>
    : public CudaSiluAndMul<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kIluvatar>>::CudaSiluAndMul;
};

}  // namespace infini::ops

#endif
