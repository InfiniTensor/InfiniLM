#ifndef INFINI_OPS_ILUVATAR_SILU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SILU_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/silu/kernel.h"

namespace infini::ops {

template <>
class Operator<Silu, Device::Type::kIluvatar>
    : public CudaSilu<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSilu<Runtime<Device::Type::kIluvatar>>::CudaSilu;
};

}  // namespace infini::ops

#endif
