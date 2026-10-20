#ifndef INFINI_OPS_NVIDIA_CONV1D_KERNEL_H_
#define INFINI_OPS_NVIDIA_CONV1D_KERNEL_H_

#include <utility>

#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv1d, Device::Type::kNvidia>
    : public CudaConv<Runtime<Device::Type::kNvidia>, Conv1d> {
 public:
  using CudaConv<Runtime<Device::Type::kNvidia>, Conv1d>::CudaConv;
};

}  // namespace infini::ops

#endif
