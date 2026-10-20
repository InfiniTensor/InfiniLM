#ifndef INFINI_OPS_NVIDIA_FILL_KERNEL_H_
#define INFINI_OPS_NVIDIA_FILL_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/fill/kernel.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kNvidia>
    : public CudaFill<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaFill<Runtime<Device::Type::kNvidia>>::CudaFill;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_FILL_KERNEL_H_
