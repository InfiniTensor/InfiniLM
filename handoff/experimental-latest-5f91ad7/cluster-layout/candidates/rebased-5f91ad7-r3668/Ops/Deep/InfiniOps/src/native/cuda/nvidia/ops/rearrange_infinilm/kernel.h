#ifndef INFINI_OPS_NVIDIA_REARRANGE_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_REARRANGE_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/rearrange_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<RearrangeInfinilm, Device::Type::kNvidia>
    : public CudaRearrangeInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaRearrangeInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaRearrangeInfinilm;
};

}  // namespace infini::ops

#endif
