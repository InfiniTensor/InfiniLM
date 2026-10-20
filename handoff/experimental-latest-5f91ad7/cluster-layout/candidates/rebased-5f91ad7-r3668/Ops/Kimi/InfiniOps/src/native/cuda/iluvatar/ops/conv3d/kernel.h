#ifndef INFINI_OPS_ILUVATAR_CONV3D_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CONV3D_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv3d, Device::Type::kIluvatar>
    : public CudaConv<Runtime<Device::Type::kIluvatar>, Conv3d> {
 public:
  using CudaConv<Runtime<Device::Type::kIluvatar>, Conv3d>::CudaConv;
};

}  // namespace infini::ops

#endif
