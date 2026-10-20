#ifndef INFINI_OPS_ILUVATAR_MUL_KERNEL_H_
#define INFINI_OPS_ILUVATAR_MUL_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/mul/kernel.h"

namespace infini::ops {

template <>
class Operator<Mul, Device::Type::kIluvatar>
    : public CudaMul<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaMul<Runtime<Device::Type::kIluvatar>>::CudaMul;
};

}  // namespace infini::ops

#endif
