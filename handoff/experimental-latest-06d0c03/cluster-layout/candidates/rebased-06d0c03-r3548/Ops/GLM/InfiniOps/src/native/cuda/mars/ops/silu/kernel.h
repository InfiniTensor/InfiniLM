#ifndef INFINI_OPS_MARS_SILU_KERNEL_H_
#define INFINI_OPS_MARS_SILU_KERNEL_H_

#include <utility>

#include "native/cuda/mars/caster.cuh"
#include "native/cuda/mars/runtime_.h"
#include "native/cuda/ops/silu/kernel.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kMars>
    : public CudaSilu<Runtime<Device::Type::kMars>> {
 public:
  using CudaSilu<Runtime<Device::Type::kMars>>::CudaSilu;
};

}  // namespace infini::ops

#endif
