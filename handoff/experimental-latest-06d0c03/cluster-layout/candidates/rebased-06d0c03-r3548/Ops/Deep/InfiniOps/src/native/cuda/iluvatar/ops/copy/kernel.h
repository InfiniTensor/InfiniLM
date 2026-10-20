#ifndef INFINI_OPS_ILUVATAR_COPY_KERNEL_H_
#define INFINI_OPS_ILUVATAR_COPY_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kIluvatar>
    : public CudaCopy<Runtime<Device::Type::kIluvatar>> {
 public:
  using CudaCopy<Runtime<Device::Type::kIluvatar>>::CudaCopy;
};

}  // namespace infini::ops

#endif
