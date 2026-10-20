#ifndef INFINI_OPS_ILUVATAR_ADD_KERNEL_H_
#define INFINI_OPS_ILUVATAR_ADD_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/add/kernel.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kIluvatar>
    : public CudaAdd<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaAdd<Runtime<Device::Type::kIluvatar>>::CudaAdd;
};

}  // namespace infini::ops

#endif
