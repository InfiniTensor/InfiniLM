#ifndef INFINI_OPS_METAX_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_METAX_SILU_AND_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kMetax>
    : public CudaSiluAndMul<Runtime<Device::Type::kMetax>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kMetax>>::CudaSiluAndMul;
};

}  // namespace infini::ops

#endif
