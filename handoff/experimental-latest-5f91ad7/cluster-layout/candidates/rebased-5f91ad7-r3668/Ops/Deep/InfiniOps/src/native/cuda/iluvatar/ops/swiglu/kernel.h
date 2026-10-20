#ifndef INFINI_OPS_ILUVATAR_SWIGLU_KERNEL_H_
#define INFINI_OPS_ILUVATAR_SWIGLU_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/swiglu/kernel.h"

namespace infini::ops {

template <>
class Operator<Swiglu, Device::Type::kIluvatar>
    : public CudaSwiglu<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaSwiglu<Runtime<Device::Type::kIluvatar>>::CudaSwiglu;
};

}  // namespace infini::ops

#endif
