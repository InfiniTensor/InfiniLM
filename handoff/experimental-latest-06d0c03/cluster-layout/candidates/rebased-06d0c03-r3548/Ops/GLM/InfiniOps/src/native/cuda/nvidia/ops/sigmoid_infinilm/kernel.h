#ifndef INFINI_OPS_NVIDIA_SIGMOID_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_SIGMOID_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/sigmoid_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<SigmoidInfinilm, Device::Type::kNvidia>
    : public CudaSigmoidInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSigmoidInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaSigmoidInfinilm;
};

}  // namespace infini::ops

#endif
