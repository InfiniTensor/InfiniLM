#ifndef INFINI_OPS_NVIDIA_SILU_AND_MUL_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_SILU_AND_MUL_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/silu_and_mul_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SiluAndMulInfinilm, Device::Type::kNvidia>
    : public CudaSiluAndMulInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSiluAndMulInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaSiluAndMulInfinilm;
};

}  // namespace infini::ops

#endif
