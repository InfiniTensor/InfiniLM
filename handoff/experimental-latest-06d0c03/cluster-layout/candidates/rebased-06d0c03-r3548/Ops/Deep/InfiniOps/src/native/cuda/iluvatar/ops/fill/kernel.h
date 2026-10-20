#ifndef INFINI_OPS_ILUVATAR_FILL_KERNEL_H_
#define INFINI_OPS_ILUVATAR_FILL_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/fill/kernel.h"

namespace infini::ops {

template <>
class Operator<Fill, Device::Type::kIluvatar>
    : public CudaFill<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaFill<Runtime<Device::Type::kIluvatar>>::CudaFill;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ILUVATAR_FILL_KERNEL_H_
