#ifndef INFINI_OPS_THEAD_SILU_KERNEL_H_
#define INFINI_OPS_THEAD_SILU_KERNEL_H_

#include <utility>

#include "native/cuda/ops/silu/kernel.h"
#include "native/cuda/thead/caster.cuh"
#include "native/cuda/thead/runtime_.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kThead>
    : public CudaSilu<Runtime<Device::Type::kThead>> {
 public:
  using CudaSilu<Runtime<Device::Type::kThead>>::CudaSilu;
};

}  // namespace infini::ops

#endif
