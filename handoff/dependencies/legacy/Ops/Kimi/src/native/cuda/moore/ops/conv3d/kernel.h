#ifndef INFINI_OPS_MOORE_CONV3D_KERNEL_H_
#define INFINI_OPS_MOORE_CONV3D_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv3d, Device::Type::kMoore>
    : public CudaConv<Runtime<Device::Type::kMoore>, Conv3d> {
 public:
  using CudaConv<Runtime<Device::Type::kMoore>, Conv3d>::CudaConv;
};

}  // namespace infini::ops

#endif
