#ifndef INFINI_OPS_NVIDIA_RELU_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_RELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/relu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ReluInfinilm, Device::Type::kNvidia>
    : public CudaReluInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaReluInfinilm<Runtime<Device::Type::kNvidia>>::CudaReluInfinilm;
};

}  // namespace infini::ops

#endif
