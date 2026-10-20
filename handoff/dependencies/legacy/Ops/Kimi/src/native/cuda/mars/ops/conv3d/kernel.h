#ifndef INFINI_OPS_MARS_CONV3D_KERNEL_H_
#define INFINI_OPS_MARS_CONV3D_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv3d, Device::Type::kMars>
    : public CudaConv<Runtime<Device::Type::kMars>, Conv3d> {
 public:
  using CudaConv<Runtime<Device::Type::kMars>, Conv3d>::CudaConv;
};

}  // namespace infini::ops

#endif
