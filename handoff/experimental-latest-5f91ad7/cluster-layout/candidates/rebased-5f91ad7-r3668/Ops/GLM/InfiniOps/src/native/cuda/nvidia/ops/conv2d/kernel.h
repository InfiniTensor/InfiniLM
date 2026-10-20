#ifndef INFINI_OPS_NVIDIA_CONV2D_KERNEL_H_
#define INFINI_OPS_NVIDIA_CONV2D_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv2d, Device::Type::kNvidia>
    : public CudaConv<Runtime<Device::Type::kNvidia>, Conv2d> {
 public:
  using CudaConv<Runtime<Device::Type::kNvidia>, Conv2d>::CudaConv;
};

}  // namespace infini::ops

#endif
