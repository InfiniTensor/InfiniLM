#ifndef INFINI_OPS_HYGON_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_HYGON_CONVOLUTION_KERNEL_H_

#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kHygon>
    : public CudaConv<Runtime<Device::Type::kHygon>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kHygon>, Convolution>::CudaConv;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_CONVOLUTION_KERNEL_H_
