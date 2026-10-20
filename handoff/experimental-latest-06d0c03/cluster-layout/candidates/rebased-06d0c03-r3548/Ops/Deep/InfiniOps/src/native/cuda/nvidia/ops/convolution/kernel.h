#ifndef INFINI_OPS_NVIDIA_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_NVIDIA_CONVOLUTION_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kNvidia>
    : public CudaConv<Runtime<Device::Type::kNvidia>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kNvidia>, Convolution>::CudaConv;

  bool capture_safe() const override { return true; }
};

}  // namespace infini::ops

#endif
