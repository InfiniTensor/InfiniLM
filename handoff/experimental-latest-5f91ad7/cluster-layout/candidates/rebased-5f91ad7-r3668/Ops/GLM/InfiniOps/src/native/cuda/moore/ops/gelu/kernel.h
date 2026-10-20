#ifndef INFINI_OPS_MOORE_GELU_KERNEL_H_
#define INFINI_OPS_MOORE_GELU_KERNEL_H_

#include <utility>

#include "native/cuda/moore/caster.cuh"
#include "native/cuda/moore/polyfills.cuh"
#include "native/cuda/moore/runtime_.h"
#include "native/cuda/ops/gelu/kernel.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kMoore>
    : public CudaGelu<Runtime<Device::Type::kMoore>> {
 public:
  using CudaGelu<Runtime<Device::Type::kMoore>>::CudaGelu;
};

}  // namespace infini::ops

#endif
