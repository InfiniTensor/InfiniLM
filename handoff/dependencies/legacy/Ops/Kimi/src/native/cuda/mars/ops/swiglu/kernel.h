#ifndef INFINI_OPS_MARS_SWIGLU_KERNEL_H_
#define INFINI_OPS_MARS_SWIGLU_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/swiglu/kernel.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kMars>
    : public CudaSwiglu<Runtime<Device::Type::kMars>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kMars>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
