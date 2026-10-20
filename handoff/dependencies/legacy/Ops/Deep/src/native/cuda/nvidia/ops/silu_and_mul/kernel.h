#ifndef INFINI_OPS_NVIDIA_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_NVIDIA_SILU_AND_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/silu_and_mul/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kNvidia>
    : public CudaSiluAndMul<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSiluAndMul<Runtime<Device::Type::kNvidia>>::CudaSiluAndMul;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
