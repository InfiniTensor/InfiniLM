#ifndef INFINI_OPS_NVIDIA_ZEROS_INFINILM_KERNEL_H_
#define INFINI_OPS_NVIDIA_ZEROS_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/zeros_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<ZerosInfinilm, Device::Type::kNvidia>
    : public CudaZerosInfinilm<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaZerosInfinilm<Runtime<Device::Type::kNvidia>>::CudaZerosInfinilm;
};

}  // namespace infini::ops

#endif
