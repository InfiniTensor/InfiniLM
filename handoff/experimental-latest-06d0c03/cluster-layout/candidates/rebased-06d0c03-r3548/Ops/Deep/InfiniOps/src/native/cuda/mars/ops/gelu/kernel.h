#ifndef INFINI_OPS_MARS_GELU_KERNEL_H_
#define INFINI_OPS_MARS_GELU_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kMars>
    : public CudaGelu<Runtime<Device::Type::kMars>> {
 public:
  using CudaGelu<Runtime<Device::Type::kMars>>::CudaGelu;
};

}  // namespace infini::ops

#endif
