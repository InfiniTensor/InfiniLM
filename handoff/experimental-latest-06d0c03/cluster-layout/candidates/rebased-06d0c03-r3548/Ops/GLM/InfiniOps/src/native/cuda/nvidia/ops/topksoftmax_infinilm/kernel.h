#ifndef INFINI_OPS_NVIDIA_TOPKSOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_TOPKSOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/topksoftmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<TopksoftmaxInfinilm, Device::Type::kNvidia>
    : public CudaTopksoftmaxInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaTopksoftmaxInfinilm<
      Runtime<Device::Type::kNvidia>>::CudaTopksoftmaxInfinilm;
};

}  // namespace infini::ops

#endif
