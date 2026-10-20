#ifndef INFINI_OPS_ILUVATAR_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CONVOLUTION_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kIluvatar>
    : public CudaConv<Runtime<Device::Type::kIluvatar>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kIluvatar>, Convolution>::CudaConv;
};

}  // namespace infini::ops

#endif
