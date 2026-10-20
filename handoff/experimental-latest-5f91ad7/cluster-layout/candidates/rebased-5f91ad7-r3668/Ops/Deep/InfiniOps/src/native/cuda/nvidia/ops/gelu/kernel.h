#ifndef INFINI_OPS_NVIDIA_GELU_KERNEL_H_
#define INFINI_OPS_NVIDIA_GELU_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kNvidia>
    : public CudaGelu<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaGelu<Runtime<Device::Type::kNvidia>>::CudaGelu;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
