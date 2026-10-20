#ifndef INFINI_OPS_NVIDIA_CONV_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_CONV_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/conv_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ConvInfinilm, Device::Type::kNvidia>
    : public CudaConvInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaConvInfinilm<Runtime<Device::Type::kNvidia>>::CudaConvInfinilm;
};

}  // namespace infini::ops

#endif
