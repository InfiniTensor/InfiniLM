#ifndef INFINI_OPS_NVIDIA_SILU_KERNEL_H_
#define INFINI_OPS_NVIDIA_SILU_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/silu/kernel.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kNvidia>
    : public CudaSilu<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSilu<Runtime<Device::Type::kNvidia>>::CudaSilu;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
