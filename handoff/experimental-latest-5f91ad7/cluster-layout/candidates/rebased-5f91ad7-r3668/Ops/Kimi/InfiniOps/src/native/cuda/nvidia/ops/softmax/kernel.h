#ifndef INFINI_OPS_NVIDIA_SOFTMAX_KERNEL_H_
#define INFINI_OPS_NVIDIA_SOFTMAX_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/softmax/kernel.h"

namespace infini::ops {

template <>
class Operator<Softmax, Device::Type::kNvidia>
    : public CudaSoftmax<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSoftmax<Runtime<Device::Type::kNvidia>>::CudaSoftmax;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
