#ifndef INFINI_OPS_METAX_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_METAX_CONVOLUTION_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kMetax>
    : public CudaConv<Runtime<Device::Type::kMetax>, Convolution> {
 public:
  using CudaConv<Runtime<Device::Type::kMetax>, Convolution>::CudaConv;
};

}  // namespace infini::ops

#endif
