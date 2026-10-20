#ifndef INFINI_OPS_METAX_CONV2D_KERNEL_H_
#define INFINI_OPS_METAX_CONV2D_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Conv2d, Device::Type::kMetax>
    : public CudaConv<Runtime<Device::Type::kMetax>, Conv2d> {
 public:
  using CudaConv<Runtime<Device::Type::kMetax>, Conv2d>::CudaConv;
};

}  // namespace infini::ops

#endif
