#ifndef INFINI_OPS_METAX_COPY_KERNEL_H_
#define INFINI_OPS_METAX_COPY_KERNEL_H_

#include <utility>

#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/copy/kernel.h"

namespace infini::ops {

template <>
class Operator<Copy, Device::Type::kMetax>
    : public CudaCopy<Runtime<Device::Type::kMetax>> {
 public:
  using CudaCopy<Runtime<Device::Type::kMetax>>::CudaCopy;
};

}  // namespace infini::ops

#endif
