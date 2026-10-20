#ifndef INFINI_OPS_NVIDIA_RELU_KERNEL_H_
#define INFINI_OPS_NVIDIA_RELU_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/relu/kernel.h"

namespace infini::ops {

template <>
class Operator<Relu, Device::Type::kNvidia>
    : public CudaRelu<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRelu<Runtime<Device::Type::kNvidia>>::CudaRelu;
};

}  // namespace infini::ops

#endif
