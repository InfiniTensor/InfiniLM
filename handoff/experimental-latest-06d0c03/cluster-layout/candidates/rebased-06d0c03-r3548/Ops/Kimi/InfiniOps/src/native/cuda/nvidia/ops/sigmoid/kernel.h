#ifndef INFINI_OPS_NVIDIA_SIGMOID_KERNEL_H_
#define INFINI_OPS_NVIDIA_SIGMOID_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/sigmoid/kernel.h"

namespace infini::ops {

template <>
class Operator<Sigmoid, Device::Type::kNvidia>
    : public CudaSigmoid<Runtime<Device::Type::kNvidia>> {
 public:
  using CudaSigmoid<Runtime<Device::Type::kNvidia>>::CudaSigmoid;
  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
