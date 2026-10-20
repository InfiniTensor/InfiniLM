#ifndef INFINI_OPS_NVIDIA_GELU_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_GELU_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/gelu_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<GeluInfinilm, Device::Type::kNvidia>
    : public CudaGeluInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaGeluInfinilm<Runtime<Device::Type::kNvidia>>::CudaGeluInfinilm;
};

}  // namespace infini::ops

#endif
